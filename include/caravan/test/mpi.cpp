/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <caravan/mpi.hpp>
#include <caravan/mpi_native.hpp>

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
            assert(topology.hostLocalRank >= 0);

            auto cartesianFuture
                = mpi.createCartesian(caravan::readyEvent(), std::vector<int>{topology.size}, std::vector<bool>{true});
            auto const cartesian = cartesianFuture.result();
            assert(cartesian.communicator != caravan::worldCommunicator);
            assert(cartesian.dimensions == std::vector<int>{topology.size});
            assert(cartesian.coordinates == std::vector<int>{topology.rank});
            assert(cartesian.periodic == std::vector<bool>{true});
            assert(cartesian.neighbors[0] == (topology.rank + topology.size - 1) % topology.size);
            assert(cartesian.neighbors[1] == (topology.rank + 1) % topology.size);

            auto first = mpi.barrier(caravan::readyEvent(), cartesian.communicator);
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
                caravan::anyMessageTag,
                cartesian.communicator);
            auto sent = mpi.send(
                caravan::readyEvent(),
                caravan::BufferLease{sentValue, sentValue.get(), sizeof(int)},
                caravan::Peer{destination},
                caravan::MessageTag{17},
                cartesian.communicator);
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
                caravan::MessageTag{18},
                cartesian.communicator);
            auto largeSend = mpi.send(
                caravan::readyEvent(),
                caravan::BufferLease{largeSendBuffer, largeSendBuffer->data(), largeSendBuffer->size()},
                caravan::Peer{destination},
                caravan::MessageTag{18},
                cartesian.communicator);
            std::array largeTransfers{largeReceive.event(), largeSend.event()};
            caravan::whenAll(largeTransfers).wait();
            assert(largeReceive.result().bytes == largeMessageBytes);
            assert(largeReceiveBuffer->front() == std::byte{42});
            assert(largeReceiveBuffer->back() == std::byte{42});

            auto reductionInput
                = std::make_shared<std::array<std::int32_t, 3>>(std::array{topology.rank + 1, 1, topology.rank});
            auto reductionOutput = std::make_shared<std::array<std::int32_t, 3>>();
            caravan::EventSource reductionReady;
            auto reduction = mpi.allReduce(
                reductionReady.event(),
                caravan::BufferLease{reductionInput, reductionInput->data(), sizeof(*reductionInput)},
                caravan::BufferLease{reductionOutput, reductionOutput->data(), sizeof(*reductionOutput)},
                caravan::ScalarType::int32,
                caravan::ReduceOperation::sum,
                cartesian.communicator);
            assert(reduction.state() == caravan::CompletionState::pending);
            reductionReady.setReady();
            assert(reduction.result().elements == reductionInput->size());
            assert((*reductionOutput)[0] == topology.size * (topology.size + 1) / 2);
            assert((*reductionOutput)[1] == topology.size);
            assert((*reductionOutput)[2] == topology.size * (topology.size - 1) / 2);

            auto nativeInput = std::make_shared<int>(topology.rank + 1);
            auto nativeOutput = std::make_shared<int>(0);
            auto nativeReduction = caravan::nativeFuture<int>(
                mpi,
                caravan::readyEvent(),
                [nativeInput, nativeOutput, communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    caravan::NativeRequestBatch batch(
                        std::vector<MPI_Request>(2, MPI_REQUEST_NULL),
                        {nativeInput, nativeOutput});
                    int error = MPI_Iallreduce(
                        nativeInput.get(),
                        nativeOutput.get(),
                        1,
                        MPI_INT,
                        MPI_SUM,
                        context.communicator(communicator),
                        &batch.requests[0]);
                    if(error == MPI_SUCCESS)
                        error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[1]);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("native MPI request start failed");
                    return batch;
                },
                [nativeOutput,
                 communicator
                 = cartesian.communicator](caravan::NativeMpiContext& context, std::span<MPI_Status const> statuses)
                {
                    int rank = -1;
                    assert(MPI_Comm_rank(context.communicator(communicator), &rank) == MPI_SUCCESS);
                    assert(rank >= 0 && statuses.size() == 2u);
                    return *nativeOutput;
                });
            assert(nativeReduction.result() == topology.size * (topology.size + 1) / 2);

            auto nativeImmediate = caravan::nativeEvent(
                mpi,
                caravan::readyEvent(),
                [](caravan::NativeMpiContext&) { return caravan::NativeRequestBatch{}; },
                [](std::span<MPI_Status const> statuses) { assert(statuses.empty()); });
            nativeImmediate.wait();

            auto nativeStartFailure = caravan::nativeEvent(
                mpi,
                caravan::readyEvent(),
                [communicator
                 = cartesian.communicator](caravan::NativeMpiContext& context) -> caravan::NativeRequestBatch
                {
                    caravan::NativeRequestBatch batch(std::vector<MPI_Request>(1, MPI_REQUEST_NULL));
                    if(MPI_Ibarrier(context.communicator(communicator), &batch.requests[0]) != MPI_SUCCESS)
                        throw std::runtime_error("native cleanup MPI_Ibarrier failed");
                    throw std::runtime_error("expected native start failure");
                },
                [](std::span<MPI_Status const>) { assert(false); });
            try
            {
                nativeStartFailure.wait();
                assert(false);
            }
            catch(std::runtime_error const&)
            {
            }

            static_cast<void>(mpi.barrier(caravan::readyEvent(), cartesian.communicator));
            auto duplicated = caravan::nativeBlockingFuture<caravan::CommunicatorId>(
                mpi,
                caravan::readyEvent(),
                [communicator = cartesian.communicator](caravan::NativeMpiContext& context)
                {
                    MPI_Comm native = MPI_COMM_NULL;
                    int const error = MPI_Comm_dup(context.communicator(communicator), &native);
                    if(error != MPI_SUCCESS)
                        throw std::runtime_error("native MPI_Comm_dup failed");
                    return context.adoptCommunicator(native);
                });
            auto const duplicatedCommunicator = duplicated.result();
            mpi.barrier(caravan::readyEvent(), duplicatedCommunicator).wait();
            mpi.destroyCommunicator(caravan::readyEvent(), duplicatedCommunicator).wait();
            caravan::nativeBlockingEvent(
                mpi,
                caravan::readyEvent(),
                [&mpi](caravan::NativeMpiContext&)
                {
                    int initialized = 0;
                    assert(MPI_Initialized(&initialized) == MPI_SUCCESS && initialized != 0);
                    auto recursive = caravan::nativeEvent(
                        mpi,
                        caravan::readyEvent(),
                        [](caravan::NativeMpiContext&) { return caravan::NativeRequestBatch{}; },
                        [](std::span<MPI_Status const>) {});
                    assert(recursive.state() == caravan::CompletionState::failed);
                    try
                    {
                        recursive.wait();
                        assert(false);
                    }
                    catch(std::logic_error const&)
                    {
                    }
                })
                .wait();

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

            mpi.destroyCommunicator(caravan::readyEvent(), cartesian.communicator).wait();

            // MpiRuntime must drain native work even when the application drops its handle.
            static_cast<void>(mpi.barrier(caravan::readyEvent()));
            return 0;
        });
}
