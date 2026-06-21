/*
  mallocMC: Memory Allocator for Many Core Architectures.
*/

#include "mallocMC/creationPolicies/OldMalloc.hpp"

#include <alpaka/alpaka.hpp>

#include <mallocMC/alignmentPolicies/Noop.hpp>
#include <mallocMC/alignmentPolicies/Shrink.hpp>
#include <mallocMC/allocator.hpp>
#include <mallocMC/creationPolicies/FlatterScatter.hpp>
#include <mallocMC/creationPolicies/GallatinCuda.hpp>
#include <mallocMC/distributionPolicies/Noop.hpp>
#include <mallocMC/oOMPolicies/ReturnNull.hpp>
#include <mallocMC/reservePoolPolicies/AlpakaBuf.hpp>
#include <mallocMC/reservePoolPolicies/Noop.hpp>

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

struct GetAvailableSlotsKernel
{
    template<typename TAcc, typename TAllocHandle, typename TSharedPtr>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, TAllocHandle allocHandle, TSharedPtr sharedPtr, std::uint32_t count)
        const
    {
        auto const [nativeId] = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        if(nativeId == 0U)
            sharedPtr[0] = static_cast<int*>(allocHandle.malloc(acc, sizeof(int) * count));
        alpaka::onAcc::syncBlockThreads(acc);

        auto const slots = allocHandle.getAvailableSlots(acc, 1U);
        for(auto [id] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{count}))
        {
            if(sharedPtr[0] != nullptr)
            {
                sharedPtr[0][id] = static_cast<int>(id);
                printf("id: %u array: %d slots %u\n", id, sharedPtr[0][id], slots);
            }
            else
            {
                printf("error: device size allocation failed");
            }
        }
        alpaka::onAcc::syncBlockThreads(acc);
        if(nativeId == 0U && sharedPtr[0] != nullptr)
            allocHandle.free(acc, sharedPtr[0]);
    }
};

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
    auto sharedPtrAcc = alpaka::onHost::alloc<int*>(devAcc, alpaka::Vec{Idx{1}});

    std::cout << "Using " << deviceSpec.getName() << " with " << alpaka::onHost::demangledName(exec) << '\n';
    constexpr auto numWorkers = 32U;

    auto frameSpec = alpaka::onHost::FrameSpec{alpaka::Vec{Idx{1}}, alpaka::Vec{Idx{numWorkers}}, exec};
    queue.enqueue(
        frameSpec,
        alpaka::KernelBundle{GetAvailableSlotsKernel{}, alloc.getAllocatorHandle(), sharedPtrAcc, numWorkers});
    alpaka::onHost::wait(queue);
    std::cout << "Slots from Host: " << alloc.getAvailableSlots(devAcc, queue, 1U) << '\n';
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

            std::cout << alpaka::onHost::demangledName<FlatterScatter<FlatterScatterHeapConfig>>() << ":\n";
            result = runExample<
                Executor,
                FlatterScatter<FlatterScatterHeapConfig>,
                mallocMC::ReservePoolPolicies::AlpakaBuf>(deviceSpec, exec);
            if(result != EXIT_SUCCESS)
                return;
            std::cout << alpaka::onHost::demangledName<Scatter<FlatterScatterHeapConfig>>() << ":\n";
            result = runExample<Executor, Scatter<FlatterScatterHeapConfig>, mallocMC::ReservePoolPolicies::AlpakaBuf>(
                deviceSpec,
                exec);
#if ALPAKA_LANG_CUDA
#    ifdef mallocMC_HAS_Gallatin_AVAILABLE
            if(result == EXIT_SUCCESS)
            {
                std::cout << alpaka::onHost::demangledName<mallocMC::CreationPolicies::GallatinCuda<>>() << ":\n";
                result = runExample<
                    Executor,
                    mallocMC::CreationPolicies::GallatinCuda<>,
                    mallocMC::ReservePoolPolicies::Noop,
                    mallocMC::AlignmentPolicies::Noop>(deviceSpec, exec);
            }
#    endif
#endif
            if(result == EXIT_SUCCESS)
            {
                std::cout << alpaka::onHost::demangledName<OldMalloc>() << ":\n";
                result = runExample<Executor, OldMalloc, mallocMC::ReservePoolPolicies::Noop>(deviceSpec, exec);
            }
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors));
    return result;
}
