/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstddef>
#include <thread>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

int main(int argc, char** argv)
{
    return caravan::MpiRuntime::run(
        argc,
        argv,
        [](caravan::MpiContext& mpi)
        {
            using Dim = alpaka::DimInt<1u>;
            using Idx = std::size_t;
#if ALPAKA_ACC_GPU_CUDA_ENABLED
            using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
            using Acc = alpaka::AccGpuHipRt<Dim, Idx>;
#else
            using Acc = alpaka::AccCpuSerial<Dim, Idx>;
#endif
            using Queue = alpaka::Queue<Acc, alpaka::NonBlocking>;

            auto const device = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
            auto const host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
            Queue queue{device};
            auto deviceValue = alpaka::allocBuf<int, Idx>(device, alpaka::Vec<Dim, Idx>{1u});
            auto hostValue = alpaka::allocBuf<int, Idx>(host, alpaka::Vec<Dim, Idx>{1u});
            int received = -1;
            hostValue[0] = mpi.topology().rank;

            bool acceleratorStarted = false;
            bool mpiStarted = false;
            bool continued = false;
            auto chain = caravan::letValue(
                caravan::alpaka::submit(
                    queue,
                    [&](Queue& nativeQueue)
                    {
                        acceleratorStarted = true;
                        alpaka::memcpy(nativeQueue, deviceValue, hostValue);
                        alpaka::memcpy(nativeQueue, hostValue, deviceValue);
                    }),
                [&]
                {
                    mpiStarted = true;
                    return caravan::whenAll(
                        caravan::mpi::send(
                            mpi,
                            caravan::BufferLease::borrowed(&hostValue[0], sizeof(int)),
                            caravan::Peer{mpi.topology().rank},
                            caravan::MessageTag{951}),
                        caravan::mpi::receive(
                            mpi,
                            caravan::BufferLease::borrowed(&received, sizeof(received)),
                            caravan::Peer{mpi.topology().rank},
                            caravan::MessageTag{951}));
                });

            assert(!acceleratorStarted && !mpiStarted);
            auto const applicationThread = std::this_thread::get_id();
            caravan::RunLoop loop;
            caravan::AsyncScope scope;
            auto completion = scope.spawn(
                caravan::then(
                    caravan::continuesOn(std::move(chain), loop.scheduler()),
                    [&](caravan::SendResult sent, caravan::ReceiveResult receivedMetadata)
                    {
                        assert(std::this_thread::get_id() == applicationThread);
                        assert(sent.bytes == sizeof(int));
                        assert(receivedMetadata.bytes == sizeof(int));
                        continued = true;
                    }));

            while(completion.state() == caravan::CompletionState::pending)
            {
                loop.runReady();
                std::this_thread::yield();
            }
            completion.wait();
            scope.join().wait();

            assert(acceleratorStarted && mpiStarted && continued);
            assert(received == mpi.topology().rank);
            return 0;
        });
}
