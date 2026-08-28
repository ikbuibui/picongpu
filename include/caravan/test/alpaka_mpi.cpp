/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstddef>
#include <thread>
#include <utility>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

namespace
{
    struct Preserve
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value) const
        {
            *value += 0;
        }
    };
} // namespace

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
            auto const one = alpaka::Vec<Dim, Idx>{1u};
            auto deviceValue = alpaka::allocBuf<int, Idx>(device, one);
            auto hostValue = alpaka::allocBuf<int, Idx>(host, one);
            int received = -1;
            hostValue[0] = mpi.topology().rank;

            bool acceleratorStarted = false;
            bool mpiStarted = false;
            bool continued = false;
            auto accelerator = caravan::alpaka::then(
                caravan::alpaka::then(
                    caravan::alpaka::submit(queue, [&](Queue&) { acceleratorStarted = true; }),
                    caravan::alpaka::copy(queue, deviceValue, hostValue, one)),
                caravan::alpaka::then(
                    caravan::alpaka::kernel<Acc>(
                        queue,
                        alpaka::WorkDivMembers<Dim, Idx>{one, one, one},
                        Preserve{},
                        alpaka::getPtrNative(deviceValue)),
                    caravan::alpaka::copy(queue, hostValue, deviceValue, one)));
            auto chain = caravan::letValue(
                std::move(accelerator),
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
