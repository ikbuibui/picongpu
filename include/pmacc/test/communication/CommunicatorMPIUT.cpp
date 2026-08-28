/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/operation/Add.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <array>
#include <stdexcept>
#include <thread>

#include <caravan/mpi.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("CommunicatorMPI consumes Caravan topology snapshots")
{
    int argc = 0;
    char** argv = nullptr;
    auto const result = caravan::MpiRuntime::run(
        argc,
        argv,
        [](caravan::MpiExecutor& mpi)
        {
            auto const topology = mpi.topology();
            auto processes = pmacc::DataSpace<TEST_DIM>::create(1);
            processes.x() = topology.size;
            pmacc::Environment<TEST_DIM>::get().initDevices(mpi, processes, pmacc::DataSpace<TEST_DIM>::create(1));
            auto& communicator = pmacc::Environment<TEST_DIM>::get().GridController().getCommunicator();
            auto progressUntil = [&communicator](auto const& completion)
            {
                while(completion.state() == caravan::CompletionState::pending)
                {
                    communicator.progressAsync();
                    std::this_thread::yield();
                }
            };

            if(communicator.getRank() != topology.rank || communicator.getSize() != topology.size
               || communicator.getHostRank() != static_cast<uint32_t>(topology.hostLocalRank))
                throw std::runtime_error("Unexpected topology snapshot");
            if(communicator.getCommunicatorId() == caravan::worldCommunicator)
                throw std::runtime_error("Cartesian communicator was not retained");
            if(communicator.getSignalCommunicatorId() == caravan::worldCommunicator
               || communicator.getSignalCommunicatorId() == communicator.getCommunicatorId())
                throw std::runtime_error("Signal communicator was not duplicated");

            std::array<std::uint32_t, 2> signalInput{static_cast<std::uint32_t>(topology.rank + 1), 1u};
            std::array<std::uint32_t, 2> signalOutput{};
            auto signalReduction = communicator.startSignalAllReduce(
                signalInput.data(),
                signalOutput.data(),
                sizeof(signalInput),
                caravan::ScalarType::uint32,
                caravan::ReduceOperation::sum);
            progressUntil(signalReduction);
            if(signalReduction.result().elements != signalInput.size()
               || signalOutput[0] != static_cast<std::uint32_t>(topology.size * (topology.size + 1) / 2)
               || signalOutput[1] != static_cast<std::uint32_t>(topology.size))
                throw std::runtime_error("Signal all-reduce adapter failed");
            auto barrier = communicator.startBarrierAsync();
            progressUntil(barrier);
            barrier.wait();

            {
                pmacc::mpi::MPIReduce reduce;
                std::uint32_t local = static_cast<std::uint32_t>(topology.rank + 1);
                std::uint32_t global = 0u;
                reduce(pmacc::math::operation::Add{}, &global, &local, 1u);
                if(global != static_cast<std::uint32_t>(topology.size * (topology.size + 1) / 2))
                    throw std::runtime_error("Caravan-backed PMacc all-reduce failed");

                global = 0u;
                reduce(pmacc::math::operation::Add{}, &global, &local, 1u, pmacc::mpi::reduceMethods::Reduce{});
                if(reduce.hasResult(pmacc::mpi::reduceMethods::Reduce{})
                   && global != static_cast<std::uint32_t>(topology.size * (topology.size + 1) / 2))
                    throw std::runtime_error("Caravan-backed PMacc root reduction failed");
            }

            for(int exchange = 1; exchange < -12 * TEST_DIM + 6 * TEST_DIM * TEST_DIM + 9; ++exchange)
            {
                auto const direction = pmacc::Mask::getRelativeDirections<TEST_DIM>(exchange);
                int expected = (topology.rank + direction.x() + topology.size) % topology.size;
                if(communicator.ExchangeTypeToRank(exchange) != expected)
                    throw std::runtime_error("Invalid periodic neighbor rank");
            }

            int sent = topology.rank;
            int received = -1;
            auto receive
                = communicator.startReceiveAsync(pmacc::LEFT, reinterpret_cast<char*>(&received), sizeof(int), 7u);
            auto send = communicator.startSendAsync(pmacc::RIGHT, reinterpret_cast<char*>(&sent), sizeof(int), 7u);
            std::array transfers{receive.event(), send.event()};
            auto transferred = caravan::whenAll(transfers);
            progressUntil(transferred);
            transferred.wait();
            if(received != (topology.rank + topology.size - 1) % topology.size || receive.result().bytes != sizeof(int)
               || send.result().bytes != sizeof(int))
                throw std::runtime_error("Caravan point-to-point adapter failed");

            pmacc::Environment<>::get().finalize();
            return 0;
        });
    REQUIRE(result == 0);
}
