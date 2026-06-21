/*
  mallocMC: Memory Allocator for Many Core Architectures.
*/

#include <alpaka/alpaka.hpp>

#include <mallocMC/alignmentPolicies/Noop.hpp>
#include <mallocMC/alignmentPolicies/Shrink.hpp>
#include <mallocMC/allocator.hpp>
#include <mallocMC/creationPolicies/FlatterScatter.hpp>
#include <mallocMC/creationPolicies/GallatinCuda.hpp>
#include <mallocMC/creationPolicies/OldMalloc.hpp>
#include <mallocMC/creationPolicies/Scatter.hpp>
#include <mallocMC/distributionPolicies/Noop.hpp>
#include <mallocMC/oOMPolicies/ReturnNull.hpp>
#include <mallocMC/reservePoolPolicies/AlpakaBuf.hpp>
#include <mallocMC/reservePoolPolicies/Noop.hpp>

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

struct VectorAddKernel
{
    template<typename TAcc, typename TAllocHandle, typename TSums>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        TAllocHandle allocHandle,
        TSums sums,
        std::uint32_t len,
        std::uint32_t count) const
    {
        for(auto [id] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{count}))
        {
            auto* a = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * len));
            auto* b = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * len));
            auto* c = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * len));
            if(a == nullptr || b == nullptr || c == nullptr)
            {
                sums[id] = -1;
                if(a != nullptr)
                    allocHandle.free(acc, a);
                if(b != nullptr)
                    allocHandle.free(acc, b);
                if(c != nullptr)
                    allocHandle.free(acc, c);
                continue;
            }

            sums[id] = 0;
            for(std::uint32_t i = 0; i < len; ++i)
            {
                a[i] = static_cast<int>(id * len + i);
                b[i] = static_cast<int>(id * len + i);
                c[i] = a[i] + b[i];
                sums[id] += c[i];
            }

            allocHandle.free(acc, a);
            allocHandle.free(acc, b);
            allocHandle.free(acc, c);
        }
    }
};

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

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
        return EXIT_SUCCESS;

    auto devAcc = devSelector.makeDevice(0);
    auto queue = devAcc.makeQueue(alpaka::queueKind::blocking);
    auto const threadsPerBlock = static_cast<std::uint32_t>(
        std::min<Idx>(Idx{32}, static_cast<Idx>(devAcc.getDeviceProperties().maxThreadsPerBlock)));
    auto const numWorkers
        = static_cast<std::uint32_t>(((localLength + threadsPerBlock - 1U) / threadsPerBlock) * threadsPerBlock);

    auto sumsAcc = alpaka::onHost::alloc<int>(devAcc, alpaka::Vec{Idx{numWorkers}});
    auto sumsHost = alpaka::onHost::allocHostLike(sumsAcc);

    Allocator alloc(devAcc, queue, 64U * 1024U * 1024U);
    std::cout << "Using " << deviceSpec.getName() << " with " << alpaka::onHost::demangledName(exec) << '\n';
    std::cout << Allocator::info("\n") << '\n';

    auto frameExtent = alpaka::Vec{Idx{threadsPerBlock}};
    auto frameSpec
        = alpaka::onHost::FrameSpec{alpaka::divCeil(alpaka::Vec{Idx{numWorkers}}, frameExtent), frameExtent, exec};
    queue.enqueue(
        frameSpec,
        alpaka::KernelBundle{VectorAddKernel{}, alloc.getAllocatorHandle(), sumsAcc, localLength, numWorkers});
    alpaka::onHost::memcpy(queue, sumsHost, sumsAcc);
    alpaka::onHost::wait(queue);

    for(Idx i = 0; i < numWorkers; ++i)
    {
        if(sumsHost[i] < 0)
            return EXIT_FAILURE;
    }

    auto const sum = std::accumulate(&sumsHost[0], &sumsHost[0] + numWorkers, std::size_t{0});
    auto const expected = static_cast<std::size_t>(numWorkers) * localLength;
    auto const gaussian = expected * (expected - 1U);
    std::cout << "The sum of the arrays on GPU is " << sum << '\n';
    std::cout << "The gaussian sum as comparison: " << gaussian << '\n';
    if(sum != gaussian)
        return EXIT_FAILURE;

    if constexpr(mallocMC::Traits<Allocator>::providesAvailableSlots)
    {
        std::cout << "there are ";
        std::cout << alloc.getAvailableSlots(devAcc, queue, 1024U * 1024U);
        std::cout << " Slots of size 1MB available\n";
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
#if ALPAKA_LANG_CUDA
#    ifdef mallocMC_HAS_Gallatin_AVAILABLE
            if(result == EXIT_SUCCESS)
            {
                result = runExample<
                    Executor,
                    mallocMC::CreationPolicies::GallatinCuda<>,
                    mallocMC::ReservePoolPolicies::Noop,
                    mallocMC::AlignmentPolicies::Noop>(deviceSpec, exec);
            }
#    endif
#endif
            if(result == EXIT_SUCCESS)
                result = runExample<Executor, OldMalloc, mallocMC::ReservePoolPolicies::Noop>(deviceSpec, exec);
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors));
    return result;
}
