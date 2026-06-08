/*
  mallocMC: Memory Allocator for Many Core Architectures.
*/

#include <alpaka/alpaka.hpp>

#include <mallocMC/alignmentPolicies/Noop.hpp>
#include <mallocMC/alignmentPolicies/Shrink.hpp>
#include <mallocMC/allocator.hpp>
#include <mallocMC/creationPolicies/FlatterScatter.hpp>
#include <mallocMC/creationPolicies/OldMalloc.hpp>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <mallocMC/distributionPolicies/Noop.hpp>
#include <mallocMC/oOMPolicies/ReturnNull.hpp>
#include <mallocMC/reservePoolPolicies/AlpakaBuf.hpp>
#include <mallocMC/reservePoolPolicies/Noop.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <type_traits>

using mallocMC::CreationPolicies::FlatterScatter;
using mallocMC::CreationPolicies::OldMalloc;
using mallocMC::CreationPolicies::Scatter;

using Idx = std::size_t;

constexpr std::uint32_t blocksize = 2U * 1024U * 1024U;
constexpr std::uint32_t pagesize = 4U * 1024U;
constexpr std::uint32_t wasteFactor = 1U;

struct FlatterScatterHeapConfig : FlatterScatter<>::Properties::HeapConfig
{
    static constexpr auto accessblocksize = blocksize;
    static constexpr auto pagesize = ::pagesize;
    static constexpr auto heapsize = 64U * 1024U * 1024U;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = wasteFactor;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};

template<typename TExecutor>
auto makeWorkDiv(auto const& devAcc, std::uint32_t numWorkers)
{
    auto threads = std::max<Idx>(
        1u,
        std::min<Idx>(static_cast<Idx>(numWorkers), devAcc.getDeviceProperties().maxThreadsPerBlock));
    auto blocks = std::max<Idx>(1u, static_cast<Idx>((numWorkers + threads - 1u) / threads));
    if constexpr(
        std::is_same_v<TExecutor, alpaka::exec::CpuSerial>
#ifndef ALPAKA_DISABLE_EXEC_CpuOmpBlocks
        || std::is_same_v<TExecutor, alpaka::exec::CpuOmpBlocks>
#endif
#ifndef ALPAKA_DISABLE_EXEC_CpuTbbBlocks
        || std::is_same_v<TExecutor, alpaka::exec::CpuTbbBlocks>
#endif
    )
    {
        blocks *= threads;
        threads = 1u;
    }
    return alpaka::onHost::ThreadSpec{alpaka::Vec{blocks}, alpaka::Vec{threads}, TExecutor{}};
}

template<
    typename TExecutor,
    typename TCreationPolicy,
    typename TReservePoolPolicy,
    typename TAlignmentPolicy = mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>
auto runExample(auto const& deviceSpec, TExecutor exec) -> int
{
    using Allocator = mallocMC::Allocator<
        TExecutor,
        TCreationPolicy,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        TReservePoolPolicy,
        TAlignmentPolicy>;

    constexpr std::uint32_t localLength = 100U;
    constexpr std::uint32_t numArrays = 32U;

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        return EXIT_SUCCESS;

    auto devAcc = devSelector.makeDevice(0);
    auto queue = devAcc.makeQueue(alpaka::queueKind::blocking);

    auto pointerExtent = alpaka::Vec{Idx{numArrays}};
    auto aPtrs = alpaka::onHost::alloc<int*>(devAcc, pointerExtent);
    auto bPtrs = alpaka::onHost::alloc<int*>(devAcc, pointerExtent);
    auto cPtrs = alpaka::onHost::alloc<int*>(devAcc, pointerExtent);
    auto sumsAcc = alpaka::onHost::alloc<int>(devAcc, pointerExtent);
    auto sumsHost = alpaka::onHost::allocHostLike(sumsAcc);

    Allocator alloc(devAcc, queue, 64U * 1024U * 1024U);
    std::cout << "Using " << deviceSpec.getName() << " with " << alpaka::onHost::demangledName(exec) << '\n';
    std::cout << Allocator::info("\n") << '\n';

    auto initKernel = [] ALPAKA_FN_ACC(auto const& acc, auto allocHandle, auto a, auto b, auto c, std::uint32_t len)
    {
        auto id = static_cast<std::uint32_t>(
            acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads)[0]);
        a[id] = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * len));
        b[id] = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * len));
        c[id] = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * len));
        for(std::uint32_t i = 0; i < len; ++i)
        {
            a[id][i] = static_cast<int>(id * len + i);
            b[id][i] = static_cast<int>(id * len + i);
        }
    };

    auto addKernel = [] ALPAKA_FN_ACC(auto const& acc, auto a, auto b, auto c, auto sums, std::uint32_t len)
    {
        auto id = static_cast<std::uint32_t>(
            acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads)[0]);
        sums[id] = 0;
        for(std::uint32_t i = 0; i < len; ++i)
        {
            c[id][i] = a[id][i] + b[id][i];
            sums[id] += c[id][i];
        }
    };

    auto freeKernel = [] ALPAKA_FN_ACC(auto const& acc, auto allocHandle, auto a, auto b, auto c)
    {
        auto id = static_cast<std::uint32_t>(
            acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads)[0]);
        allocHandle.free(acc, a[id]);
        allocHandle.free(acc, b[id]);
        allocHandle.free(acc, c[id]);
    };

    auto workDiv = makeWorkDiv<TExecutor>(devAcc, numArrays);
    queue.enqueue(
        workDiv,
        alpaka::KernelBundle{initKernel, alloc.getAllocatorHandle(), aPtrs, bPtrs, cPtrs, localLength});
    queue.enqueue(workDiv, alpaka::KernelBundle{addKernel, aPtrs, bPtrs, cPtrs, sumsAcc, localLength});
    alpaka::onHost::memcpy(queue, sumsHost, sumsAcc);
    alpaka::onHost::wait(queue);

    auto const sum = std::accumulate(&sumsHost[0], &sumsHost[0] + numArrays, std::size_t{0});
    auto const n = static_cast<std::size_t>(numArrays) * localLength;
    auto const expected = n * (n - 1);
    std::cout << "sum=" << sum << " expected=" << expected << '\n';
    if(sum != expected)
        return EXIT_FAILURE;

    if constexpr(mallocMC::Traits<Allocator>::providesAvailableSlots)
        std::cout << "available 1MB slots: " << alloc.getAvailableSlots(devAcc, queue, 1024U * 1024U) << '\n';

    queue.enqueue(workDiv, alpaka::KernelBundle{freeKernel, alloc.getAllocatorHandle(), aPtrs, bPtrs, cPtrs});
    alpaka::onHost::wait(queue);
    return EXIT_SUCCESS;
}

auto main() -> int
{
    int result = EXIT_SUCCESS;
    alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            auto const deviceSpec = backend[alpaka::object::deviceSpec];
            auto const exec = backend[alpaka::object::exec];
            using Executor = std::decay_t<decltype(exec)>;
            if(result != EXIT_SUCCESS)
                return;

            result = runExample<
                Executor,
                FlatterScatter<FlatterScatterHeapConfig>,
                mallocMC::ReservePoolPolicies::AlpakaBuf>(deviceSpec, exec);
            if(result != EXIT_SUCCESS)
                return;
            result = runExample<Executor, Scatter<FlatterScatterHeapConfig>, mallocMC::ReservePoolPolicies::AlpakaBuf>(
                deviceSpec,
                exec);
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    ifdef mallocMC_HAS_Gallatin_AVAILABLE
            if(result == EXIT_SUCCESS)
            {
                result = EXIT_SUCCESS;
            }
#    endif
#endif
            if(result == EXIT_SUCCESS)
                result = runExample<Executor, OldMalloc, mallocMC::ReservePoolPolicies::Noop>(deviceSpec, exec);
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors));
    return result;
}
