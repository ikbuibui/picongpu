/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <array>
#include <stdexcept>

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

            if(communicator.getRank() != topology.rank || communicator.getSize() != topology.size
               || communicator.getHostRank() != static_cast<uint32_t>(topology.hostLocalRank))
                throw std::runtime_error("Unexpected topology snapshot");
            if(communicator.getCommunicatorId() == caravan::worldCommunicator)
                throw std::runtime_error("Cartesian communicator was not retained");
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
            caravan::whenAll(transfers).wait();
            if(received != (topology.rank + topology.size - 1) % topology.size || receive.result().bytes != sizeof(int)
               || send.result().bytes != sizeof(int))
                throw std::runtime_error("Caravan point-to-point adapter failed");

            pmacc::Environment<>::get().finalize();
            return 0;
        });
    REQUIRE(result == 0);
}
