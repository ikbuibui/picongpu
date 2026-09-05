/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
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

    {
        caravan::MpiExternalRuntime runtime;
        auto& context = runtime.context();
        auto const owner = std::this_thread::get_id();
        std::size_t completed = 0u;
        caravan::AsyncScope scope;
        std::array<caravan::Event, 100u> events;
        for(auto& event : events)
            event = scope.spawn(
                caravan::then(
                    caravan::mpi::invoke(
                        context,
                        [owner](caravan::NativeMpiContext&) { assert(std::this_thread::get_id() == owner); }),
                    [&] { ++completed; }));

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

    assert(MPI_Finalize() == MPI_SUCCESS);
}
