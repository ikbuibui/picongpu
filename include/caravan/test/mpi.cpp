/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <caravan/mpi.hpp>

int main(int argc, char** argv)
{
    return caravan::MpiRuntime::run(
        argc,
        argv,
        [](caravan::MpiExecutor& mpi)
        {
            auto const topology = mpi.topology();
            assert(topology.size > 0);
            assert(topology.rank >= 0 && topology.rank < topology.size);

            auto first = mpi.barrier(caravan::readyEvent());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            first.wait();

            caravan::EventSource dependency;
            auto second = mpi.barrier(dependency.event());
            assert(second.state() == caravan::CompletionState::pending);
            dependency.setReady();
            second.wait();

            int const destination = (topology.rank + 1) % topology.size;
            int const source = (topology.rank + topology.size - 1) % topology.size;
            auto sentValue = std::make_shared<int>(topology.rank);
            auto receivedValues = std::make_shared<std::array<int, 2>>(std::array{-1, -1});
            auto received = mpi.receive(
                caravan::readyEvent(),
                caravan::BufferLease{receivedValues, receivedValues->data(), sizeof(*receivedValues)},
                caravan::anyPeer,
                caravan::anyMessageTag);
            auto sent = mpi.send(
                caravan::readyEvent(),
                caravan::BufferLease{sentValue, sentValue.get(), sizeof(int)},
                caravan::Peer{destination},
                caravan::MessageTag{17});
            std::array transfers{received.event(), sent.event()};
            caravan::whenAll(transfers).wait();
            assert(receivedValues->front() == source);
            assert(receivedValues->back() == -1);
            assert(sent.result().bytes == sizeof(int));
            assert(received.result().source.value == source);
            assert(received.result().tag.value == 17);
            assert(received.result().bytes == sizeof(int));

            constexpr std::size_t largeMessageBytes = 1024u * 1024u;
            auto largeSendBuffer = std::make_shared<std::vector<std::byte>>(largeMessageBytes, std::byte{42});
            auto largeReceiveBuffer = std::make_shared<std::vector<std::byte>>(largeMessageBytes);
            auto largeReceive = mpi.receive(
                caravan::readyEvent(),
                caravan::BufferLease{largeReceiveBuffer, largeReceiveBuffer->data(), largeReceiveBuffer->size()},
                caravan::Peer{source},
                caravan::MessageTag{18});
            auto largeSend = mpi.send(
                caravan::readyEvent(),
                caravan::BufferLease{largeSendBuffer, largeSendBuffer->data(), largeSendBuffer->size()},
                caravan::Peer{destination},
                caravan::MessageTag{18});
            std::array largeTransfers{largeReceive.event(), largeSend.event()};
            caravan::whenAll(largeTransfers).wait();
            assert(largeReceive.result().bytes == largeMessageBytes);
            assert(largeReceiveBuffer->front() == std::byte{42});
            assert(largeReceiveBuffer->back() == std::byte{42});

            auto invalid = mpi.send(
                caravan::readyEvent(),
                caravan::BufferLease{std::shared_ptr<void>{}, nullptr, 1u},
                caravan::Peer{destination},
                caravan::MessageTag{19});
            try
            {
                static_cast<void>(invalid.result());
                assert(false);
            }
            catch(std::invalid_argument const&)
            {
            }

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
