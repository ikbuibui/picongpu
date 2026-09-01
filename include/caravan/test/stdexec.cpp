/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstddef>
#include <latch>
#include <thread>
#include <utility>

#include <caravan/alpaka.hpp>
#include <caravan/mpi.hpp>
#include <caravan/stdexec.hpp>
#include <exec/async_scope.hpp>
#include <stdexec/execution.hpp>

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
            namespace ex = stdexec;
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
            Queue queue{device};
            auto const one = alpaka::Vec<Dim, Idx>{1u};
            auto value = alpaka::allocBuf<int, Idx>(device, one);
            int sent = mpi.topology().rank;

            stdexec::run_loop controlLoop;
            std::thread::id controlThread;
            auto chain = caravan::stdexecInterop::adapt(
                             caravan::alpaka::kernel<Acc>(
                                 queue,
                                 alpaka::WorkDivMembers<Dim, Idx>{one, one, one},
                                 Preserve{},
                                 alpaka::getPtrNative(value)))
                         | ex::let_value(
                             [&]
                             {
                                 return caravan::stdexecInterop::adapt(
                                     caravan::mpi::send(
                                         mpi,
                                         caravan::BufferLease::borrowed(&sent, sizeof(sent)),
                                         caravan::Peer{mpi.topology().rank},
                                         caravan::MessageTag{952}));
                             })
                         | ex::continues_on(controlLoop.get_scheduler())
                         | ex::then(
                             [&](caravan::SendResult result)
                             {
                                 assert(std::this_thread::get_id() == controlThread);
                                 return result.bytes;
                             });

            static_assert(ex::sender<decltype(chain)>);
#if ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED
            // Translation-only gate: NVIDIA/stdexec async_scope does not instantiate with nvcc 13.3.
            return 0;
#else
            int received = -1;
            std::latch running{1};
            std::jthread control(
                [&]
                {
                    controlThread = std::this_thread::get_id();
                    running.count_down();
                    controlLoop.run();
                });
            running.wait();

            exec::async_scope scope;
            auto incoming = scope.spawn_future(
                caravan::stdexecInterop::adapt(
                    caravan::mpi::receive(
                        mpi,
                        caravan::BufferLease::borrowed(&received, sizeof(received)),
                        caravan::Peer{mpi.topology().rank},
                        caravan::MessageTag{952})));
            auto result = scope.spawn_future(std::move(chain));
            auto sentBytes = ex::sync_wait(std::move(result));
            auto receiveResult = ex::sync_wait(std::move(incoming));
            ex::sync_wait(scope.on_empty());
            controlLoop.finish();

            assert(std::get<0>(*sentBytes) == sizeof(sent));
            assert(std::get<0>(*receiveResult).bytes == sizeof(received));
            assert(received == sent);
            return 0;
#endif
        });
}
