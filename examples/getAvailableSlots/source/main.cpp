/*
  mallocMC: Memory Allocator for Many Core Architectures.
*/

#include "mallocMC/creationPolicies/OldMalloc.hpp"

#include <alpaka/alpaka.hpp>

#include <mallocMC/alignmentPolicies/Noop.hpp>
#include <mallocMC/alignmentPolicies/Shrink.hpp>
#include <mallocMC/allocator.hpp>
#include <mallocMC/creationPolicies/FlatterScatter.hpp>
#include <mallocMC/distributionPolicies/Noop.hpp>
#include <mallocMC/oOMPolicies/ReturnNull.hpp>
#include <mallocMC/reservePoolPolicies/AlpakaBuf.hpp>
#include <mallocMC/reservePoolPolicies/Noop.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

struct AlignmentConfig
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
    typename TAlignmentPolicy = mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>
auto runExample(auto const& deviceSpec, TExecutor exec) -> int
{
    using Allocator = mallocMC::Allocator<
        TExecutor,
        TCreationPolicy,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        TReservePoolPolicy,
        TAlignmentPolicy>;

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        return EXIT_SUCCESS;

    auto devAcc = devSelector.makeDevice(0);
    auto queue = devAcc.makeQueue(alpaka::queueKind::blocking);
    Allocator alloc(devAcc, queue, 64U * 1024U * 1024U);

    auto resultAcc = alpaka::onHost::alloc<int>(devAcc, alpaka::Vec{Idx{32}});
    auto resultHost = alpaka::onHost::allocHostLike(resultAcc);

    auto workDiv = makeWorkDiv<TExecutor>(devAcc, 32U);
    auto kernel = [] ALPAKA_FN_ACC(auto const& acc, auto allocHandle, auto out)
    {
        auto id = static_cast<std::uint32_t>(
            acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads)[0]);
        auto ptr = static_cast<int*>(allocHandle.malloc(acc, sizeof(int)));
        out[id] = (ptr != nullptr) ? static_cast<int>(id) : -1;
        if(ptr != nullptr)
            allocHandle.free(acc, ptr);
    };

    auto const before = alloc.getAvailableSlots(devAcc, queue, 1U);
    queue.enqueue(workDiv, alpaka::KernelBundle{kernel, alloc.getAllocatorHandle(), resultAcc});
    alpaka::onHost::memcpy(queue, resultHost, resultAcc);
    alpaka::onHost::wait(queue);
    auto const after = alloc.getAvailableSlots(devAcc, queue, 1U);

    std::cout << "Using " << deviceSpec.getName() << " with " << alpaka::onHost::demangledName(exec) << '\n';
    std::cout << "slots before=" << before << " after=" << after << '\n';

    for(Idx i = 0; i < 32u; ++i)
    {
        if(resultHost[i] < 0)
            return EXIT_FAILURE;
    }
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
