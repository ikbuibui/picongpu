/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <array>
#include <cassert>
#include <cstddef>
#include <thread>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_FUNNELED);
    MPI_Errhandler originalErrorHandler = MPI_ERRHANDLER_NULL;
    assert(MPI_Comm_get_errhandler(MPI_COMM_WORLD, &originalErrorHandler) == MPI_SUCCESS);

    {
        caravan::MpiExternalRuntime runtime;
        auto& context = runtime.context();
        auto const owner = std::this_thread::get_id();
        std::size_t completed = 0u;
        caravan::AsyncScope scope;
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
    assert(MPI_Comm_get_errhandler(MPI_COMM_WORLD, &restoredErrorHandler) == MPI_SUCCESS);
    assert(restoredErrorHandler == originalErrorHandler);
    assert(MPI_Errhandler_free(&originalErrorHandler) == MPI_SUCCESS);
    assert(MPI_Errhandler_free(&restoredErrorHandler) == MPI_SUCCESS);
    assert(MPI_Finalize() == MPI_SUCCESS);
}
