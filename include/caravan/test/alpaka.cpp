/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#include <alpaka/alpaka.hpp>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

namespace
{
    struct Increment
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value) const
        {
            ++*value;
        }
    };
} // namespace

int main()
{
    assert(!caravan::alpaka::isCompletionCallback());
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
    Queue secondQueue{device};
    auto const one = alpaka::Vec<Dim, Idx>{1u};
    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{one, one, one};
    auto deviceValue = alpaka::allocBuf<int, Idx>(device, one);
    auto hostValue = alpaka::allocBuf<int, Idx>(host, one);
    hostValue[0] = 41;

    bool submitted = false;
    auto retained = std::make_shared<int>(7);
    std::weak_ptr<int> retainedObserver = retained;
    auto sender = caravan::alpaka::sequence(
        caravan::alpaka::sequence(
            caravan::alpaka::sequence(
                caravan::alpaka::fill(queue, caravan::alpaka::OwnedView{deviceValue, retained}, 0u),
                caravan::alpaka::copy(
                    queue,
                    caravan::alpaka::OwnedView{deviceValue, retained},
                    caravan::alpaka::OwnedView{hostValue, retained},
                    one)),
            caravan::alpaka::kernel<Acc>(
                queue,
                workDiv,
                Increment{},
                caravan::alpaka::retain(
                    alpaka::getPtrNative(deviceValue),
                    caravan::alpaka::OwnedView{deviceValue, retained}))),
        caravan::alpaka::sequence(
            caravan::alpaka::copy(
                queue,
                caravan::alpaka::OwnedView{hostValue, retained},
                caravan::alpaka::OwnedView{deviceValue, retained},
                one),
            caravan::alpaka::submit(queue, [&](Queue&) { submitted = true; })));

    static_assert(caravan::Sender<decltype(sender)>);
    retained.reset();
    assert(!submitted);
    assert(!retainedObserver.expired());

    caravan::RunLoop loop;
    caravan::AsyncScope scope;
    std::thread::id continuationThread;
    auto completion = scope.spawn(
        caravan::then(
            caravan::continuesOn(std::move(sender), loop.scheduler()),
            [&]
            {
                assert(!caravan::alpaka::isCompletionCallback());
                continuationThread = std::this_thread::get_id();
            }));
    assert(submitted);

    while(completion.state() == caravan::CompletionState::pending)
    {
        loop.runReady();
        std::this_thread::yield();
    }
    completion.wait();
    assert(hostValue[0] == 42);
    assert(continuationThread == std::this_thread::get_id());
    assert(retainedObserver.expired());

    // A queue change is lowered to an alpaka event/native wait, not host completion between these copies.
    auto crossInput = alpaka::allocBuf<int, Idx>(host, one);
    auto crossOutput = alpaka::allocBuf<int, Idx>(host, one);
    auto crossDevice = alpaka::allocBuf<int, Idx>(device, one);
    crossInput[0] = 73;
    crossOutput[0] = 0;
    scope
        .spawn(
            caravan::alpaka::sequence(
                caravan::alpaka::copy(queue, crossDevice, crossInput, one),
                caravan::alpaka::copy(secondQueue, crossOutput, crossDevice, one)))
        .wait();
    assert(crossOutput[0] == 73);

    auto deviceSize = alpaka::allocBuf<std::size_t, Idx>(device, one);
    auto hostSize = alpaka::allocBuf<std::size_t, Idx>(host, one);
    auto sizeInput = alpaka::allocBuf<std::size_t, Idx>(host, one);
    sizeInput[0] = 123u;
    scope
        .spawn(
            caravan::alpaka::sequence(
                caravan::alpaka::copy(queue, deviceSize, sizeInput, one),
                caravan::alpaka::size(queue, hostSize, deviceSize)))
        .wait();
    assert(hostSize[0] == 123u);

    bool callbackContextObserved = false;
    scope
        .spawn(
            caravan::then(
                caravan::alpaka::submit(queue, [](Queue&) {}),
                [&] { callbackContextObserved = caravan::alpaka::isCompletionCallback(); }))
        .wait();
    assert(!callbackContextObserved);

    // Supported alpaka queues accept concurrent starts; Caravan adds no submission thread or serialization layer.
    std::atomic<unsigned> callbacks = 0u;
    std::vector<std::thread> submitters;
    for(unsigned i = 0u; i < 8u; ++i)
        submitters.emplace_back(
            [&]
            {
                scope.spawn(
                    caravan::alpaka::submit(
                        queue,
                        [&](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [&] { ++callbacks; }); }));
            });
    for(auto& thread : submitters)
        thread.join();

    auto expectFailed = [](caravan::Event const& event)
    {
        try
        {
            event.wait();
            assert(false);
        }
        catch(std::exception const&)
        {
        }
        assert(event.state() == caravan::CompletionState::failed);
    };

    auto failed
        = scope.spawn(caravan::alpaka::submit(queue, [](Queue&) { throw std::runtime_error("submission failed"); }));
    expectFailed(failed);

    // Submission cleanup must not wait on the alpaka callback/completion path that started the successor.
    auto reentrantFailure = scope.spawn(
        caravan::letValue(
            caravan::alpaka::submit(queue, [](Queue&) {}),
            [&]
            {
                return caravan::alpaka::submit(queue, [](Queue&) { throw std::runtime_error("reentrant failure"); });
            }));
    expectFailed(reentrantFailure);

#if !ALPAKA_ACC_GPU_CUDA_ENABLED && !ALPAKA_ACC_GPU_HIP_ENABLED
    // CPU queue synchronization must surface exceptions from asynchronously executed tasks.
    auto executionFailure = scope.spawn(
        caravan::alpaka::submit(
            queue,
            [](Queue& nativeQueue)
            { alpaka::enqueue(nativeQueue, [] { throw std::runtime_error("execution failed"); }); }));
    expectFailed(executionFailure);
#endif

    // No external owner: retained allocations are reclaimed only after terminal synchronization, off backend
    // callbacks.
    scope.spawn(caravan::alpaka::fill(queue, alpaka::allocBuf<int, Idx>(device, one), 0u)).wait();

    caravan::RunLoop stoppedLoop;
    auto stoppedScheduler = stoppedLoop.scheduler();
    stoppedLoop.finish();
    auto failedTransfer = scope.spawn(
        caravan::continuesOn(
            caravan::alpaka::fill(queue, alpaka::allocBuf<int, Idx>(device, one), 0u),
            stoppedScheduler));
    expectFailed(failedTransfer);

    scope.join().wait();
    assert(callbacks == 8u);
}
