/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <span>
#include <string_view>
#include <thread>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace
{
    thread_local int allocationsUntilFailure = -1;
    thread_local bool countLargeAllocations = false;
    thread_local std::size_t largeAllocations = 0u;
    bool injectedFailure = false;
    std::weak_ptr<int> liveOwner;
    MPI_Request liveRequest = MPI_REQUEST_NULL;
} // namespace

void* operator new(std::size_t bytes)
{
    if(allocationsUntilFailure >= 0 && allocationsUntilFailure-- == 0)
    {
        injectedFailure = true;
        throw std::bad_alloc{};
    }
    if(countLargeAllocations && bytes > 4096u)
        ++largeAllocations;
    if(auto* memory = std::malloc(bytes == 0u ? 1u : bytes))
        return memory;
    throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}

// Interpose only in this test: verify fail-fast precedes owner destruction, then clean
// up the unmatched receive so mpiexec observes a normal, finalized process exit.
extern "C" int MPI_Abort(MPI_Comm communicator, int error)
{
    if(!injectedFailure || liveOwner.expired() || liveRequest == MPI_REQUEST_NULL || communicator != MPI_COMM_WORLD
       || error != MPI_ERR_OTHER)
        std::_Exit(EXIT_FAILURE);
    if(MPI_Cancel(&liveRequest) != MPI_SUCCESS || MPI_Wait(&liveRequest, MPI_STATUS_IGNORE) != MPI_SUCCESS
       || MPI_Finalize() != MPI_SUCCESS)
        std::_Exit(EXIT_FAILURE);
    std::_Exit(EXIT_SUCCESS);
}

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    if(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED)
        return EXIT_FAILURE;
    MPI_Errhandler originalErrorHandler = MPI_ERRHANDLER_NULL;
    if(MPI_Comm_get_errhandler(MPI_COMM_WORLD, &originalErrorHandler) != MPI_SUCCESS)
        return EXIT_FAILURE;

    {
        caravan::MpiExternalRuntime runtime;
        auto& context = runtime.context();
        auto const owner = std::this_thread::get_id();
        std::size_t completed = 0u;
        caravan::AsyncScope scope;

        if(argc > 1)
        {
            std::set_terminate([] { std::_Exit(EXIT_FAILURE); });
            scope.spawn(
                caravan::mpi::request<void>(
                    context,
                    [](caravan::NativeMpiContext& native)
                    {
                        // Only the native batch retains this owner after the start hook returns.
                        auto storage = std::make_shared<int>(42);
                        liveOwner = storage;
                        caravan::NativeRequestBatch batch({MPI_REQUEST_NULL}, {storage});
                        if(MPI_Irecv(
                               storage.get(),
                               1,
                               MPI_INT,
                               0,
                               71,
                               native.communicator(caravan::worldCommunicator),
                               &batch.requests[0])
                           != MPI_SUCCESS)
                            std::_Exit(EXIT_FAILURE);
                        liveRequest = batch.requests[0];
                        return batch;
                    },
                    [](std::span<MPI_Status const>) { std::_Exit(EXIT_FAILURE); }));
            scope.spawn(
                caravan::mpi::invoke(
                    context,
                    [&](caravan::NativeMpiContext&)
                    {
                        // Request registration is done; fail either progress scratch-vector allocation.
                        allocationsUntilFailure = std::string_view(argv[1]) == "--fatal-statuses" ? 1 : 0;
                    }));
            runtime.progress();
            std::_Exit(EXIT_FAILURE);
        }
        std::array<caravan::Event, 100u> events;
        for(auto& event : events)
            event = scope.spawn(
                caravan::mpi::invoke(
                    context,
                    [owner](caravan::NativeMpiContext&) { assert(std::this_thread::get_id() == owner); })
                | caravan::then([&] { ++completed; }));

        assert(completed == 0u);
        assert(runtime.progress());
        assert(completed > 0u && completed < events.size());
        auto all = caravan::whenAll(events);
        while(all.state() == caravan::CompletionState::pending)
            runtime.progress();
        all.wait();

        // Queue a burst of unmatched receives. Tracking arrays must grow geometrically,
        // not reallocate for every request. Small per-operation allocations are excluded.
        constexpr std::size_t burstSize = 512u;
        std::array<int, burstSize> received{};
        std::array<caravan::Event, 2u * burstSize> burstEvents;
        auto const rank = context.topology().rank;
        for(std::size_t i = 0u; i < burstSize; ++i)
            burstEvents[i] = scope.spawn(
                caravan::mpi::receive(
                    context,
                    std::as_writable_bytes(std::span{&received[i], 1}),
                    caravan::Peer{rank},
                    caravan::MessageTag{72}));
        bool allStarted = false;
        scope.spawn(caravan::mpi::invoke(context, [&](caravan::NativeMpiContext&) { allStarted = true; }));
        countLargeAllocations = true;
        while(!allStarted)
            runtime.progress();
        countLargeAllocations = false;
        assert(largeAllocations < burstSize / 8u);

        int const sent = 42;
        for(std::size_t i = 0u; i < burstSize; ++i)
            burstEvents[burstSize + i] = scope.spawn(
                caravan::mpi::send(
                    context,
                    std::as_bytes(std::span{&sent, 1}),
                    caravan::Peer{rank},
                    caravan::MessageTag{72}));
        auto burst = caravan::whenAll(burstEvents);
        while(burst.state() == caravan::CompletionState::pending)
            runtime.progress();
        burst.wait();
        for(int value : received)
            assert(value == sent);

        auto barrier = scope.spawn(caravan::mpi::barrier(context));
        while(barrier.state() == caravan::CompletionState::pending)
            runtime.progress();
        barrier.wait();
        scope.join().wait();

        runtime.requestShutdown();
        while(runtime.progress())
        {
        }
    }

    MPI_Errhandler restoredErrorHandler = MPI_ERRHANDLER_NULL;
    if(MPI_Comm_get_errhandler(MPI_COMM_WORLD, &restoredErrorHandler) != MPI_SUCCESS
       || restoredErrorHandler != originalErrorHandler || MPI_Errhandler_free(&originalErrorHandler) != MPI_SUCCESS
       || MPI_Errhandler_free(&restoredErrorHandler) != MPI_SUCCESS || MPI_Finalize() != MPI_SUCCESS)
        return EXIT_FAILURE;
}
