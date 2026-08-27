/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <thread>

#include <caravan/mpi.hpp>

int main(int argc, char** argv)
{
    return caravan::MpiRuntime::run(
        argc,
        argv,
        [](caravan::MpiExecutor& mpi)
        {
            auto first = mpi.barrier(caravan::readyEvent());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            first.wait();

            caravan::EventSource dependency;
            auto second = mpi.barrier(dependency.event());
            assert(second.state() == caravan::CompletionState::pending);
            dependency.setReady();
            second.wait();

            caravan::EventSource failedDependency;
            auto failed = mpi.barrier(failedDependency.event());
            failedDependency.setFailed(std::make_exception_ptr(std::runtime_error("expected")));
            try
            {
                failed.wait();
                assert(false);
            }
            catch(std::runtime_error const&)
            {
            }

            // MpiRuntime must drain native work even when the application drops its handle.
            static_cast<void>(mpi.barrier(caravan::readyEvent()));
            return 0;
        });
}
