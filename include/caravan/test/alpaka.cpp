/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
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

    struct Add
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value, int const* other) const
        {
            *value += *other;
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
    auto sender
        = caravan::alpaka::fill(queue, caravan::Retained{deviceValue, retained}, 0u)
          | caravan::alpaka::sequence(
              caravan::alpaka::copy(
                  queue,
                  caravan::Retained{deviceValue, retained},
                  caravan::Retained{hostValue, retained},
                  one))
          | caravan::alpaka::sequence(
              caravan::alpaka::kernel<Acc>(
                  queue,
                  workDiv,
                  Increment{},
                  caravan::retain(alpaka::getPtrNative(deviceValue), caravan::Retained{deviceValue, retained})))
          | caravan::alpaka::sequence(
              caravan::alpaka::copy(
                  queue,
                  caravan::Retained{hostValue, retained},
                  caravan::Retained{deviceValue, retained},
                  one))
          | caravan::alpaka::sequence(
              caravan::alpaka::submit(
                  queue,
                  [&, value = std::make_unique<int>(7)](Queue&)
                  {
                      assert(*value == 7);
                      submitted = true;
                  }));

    static_assert(caravan::Sender<decltype(sender)>);
    retained.reset();
    assert(!submitted);
    assert(!retainedObserver.expired());

    caravan::RunLoop loop;
    caravan::AsyncScope scope;

    // Managed operations remain bound to their device context across a host scheduler transfer.
    caravan::alpaka::Context<Acc> context{device};
    auto managedInput = alpaka::allocBuf<int, Idx>(host, one);
    auto managedOutput = alpaka::allocBuf<int, Idx>(host, one);
    auto managedOtherOutput = alpaka::allocBuf<int, Idx>(host, one);
    auto managedValue = alpaka::allocBuf<int, Idx>(device, one);
    auto managedOther = alpaka::allocBuf<int, Idx>(device, one);
    managedInput[0] = 8;
    managedOutput[0] = 0;
    managedOtherOutput[0] = 0;
    bool changedScheduler = false;
    auto managedCompletion = scope.spawn(
        caravan::alpaka::withDevice(
            context,
            caravan::startsOn(
                loop.scheduler(),
                caravan::whenAll(
                    caravan::alpaka::copy(managedValue, managedInput, one),
                    caravan::alpaka::fill(managedOther, 0u))
                    | caravan::alpaka::sequence(
                        caravan::whenAll(
                            caravan::alpaka::kernel<Acc>(workDiv, Increment{}, alpaka::getPtrNative(managedValue)),
                            caravan::alpaka::kernel<Acc>(workDiv, Increment{}, alpaka::getPtrNative(managedOther))))
                    | caravan::alpaka::sequence(caravan::alpaka::copy(managedOutput, managedValue, one))
                    | caravan::continuesOn(loop.scheduler())
                    | caravan::letValue(
                        [&]
                        {
                            changedScheduler = true;
                            return caravan::alpaka::copy(managedOtherOutput, managedOther, one);
                        }))));
    while(managedCompletion.state() == caravan::CompletionState::pending)
    {
        loop.runReady();
        std::this_thread::yield();
    }
    managedCompletion.wait();
    assert(changedScheduler && managedOutput[0] == 9 && managedOtherOutput[0] == 1);

    std::thread::id continuationThread;
    auto completion = scope.spawn(
        std::move(sender) | caravan::continuesOn(loop.scheduler())
        | caravan::then(
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

    // A non-owning view and a raw kernel argument survive their original allocation handle's scope.
    auto localStep = [&]
    {
        auto allocation = alpaka::allocBuf<int, Idx>(device, one);
        auto view = caravan::retain(alpaka::createView(device, alpaka::getPtrNative(allocation), one), allocation);
        return caravan::alpaka::fill(queue, view, 0u)
               | caravan::alpaka::sequence(
                   caravan::alpaka::kernel<Acc>(
                       queue,
                       workDiv,
                       Increment{},
                       caravan::retain(alpaka::getPtrNative(allocation), view)))
               | caravan::alpaka::sequence(caravan::alpaka::copy(queue, hostValue, view, one));
    }();
    caravan::syncWait(std::move(localStep));
    assert(hostValue[0] == 1);

    // A queue change is lowered to an alpaka event/native wait, not host completion between these copies.
    auto crossInput = alpaka::allocBuf<int, Idx>(host, one);
    auto crossOutput = alpaka::allocBuf<int, Idx>(host, one);
    auto crossDevice = alpaka::allocBuf<int, Idx>(device, one);
    crossInput[0] = 73;
    crossOutput[0] = 0;
    scope
        .spawn(
            caravan::alpaka::copy(queue, crossDevice, crossInput, one)
            | caravan::alpaka::sequence(caravan::alpaka::copy(secondQueue, crossOutput, crossDevice, one)))
        .wait();
    assert(crossOutput[0] == 73);

    // Explicit graph edges lower to native queue ordering without adding the false A -> D dependency.
    std::atomic<unsigned> graphA = 0u, graphB = 0u, graphC = 0u, graphD = 0u;
    auto graphNodeA = caravan::node<"a">(
        caravan::alpaka::submit(queue, [&](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [&] { ++graphA; }); })
        | caravan::alpaka::sequence(
            caravan::alpaka::submit(
                queue,
                [&](Queue& nativeQueue)
                {
                    alpaka::enqueue(
                        nativeQueue,
                        [&]
                        {
                            assert(graphA == 1u);
                            ++graphA;
                        });
                })));
    auto graphNodeB = caravan::node<"b">(caravan::alpaka::submit(
        secondQueue,
        [&](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [&] { ++graphB; }); }));
    auto graphNodeC = caravan::node<"c">(
        caravan::alpaka::submit(
            queue,
            [&](Queue& nativeQueue)
            {
                alpaka::enqueue(
                    nativeQueue,
                    [&]
                    {
                        assert(graphA == 2u && graphB == 1u);
                        ++graphC;
                    });
            })
            | caravan::alpaka::sequence(
                caravan::alpaka::submit(
                    queue,
                    [&](Queue& nativeQueue)
                    {
                        alpaka::enqueue(
                            nativeQueue,
                            [&]
                            {
                                assert(graphC == 1u);
                                ++graphC;
                            });
                    })),
        caravan::after(graphNodeA, graphNodeB));
    auto graphNodeD = caravan::node<"d">(
        caravan::alpaka::submit(
            secondQueue,
            [&](Queue& nativeQueue)
            {
                alpaka::enqueue(
                    nativeQueue,
                    [&]
                    {
                        assert(graphB == 1u);
                        ++graphD;
                    });
            }),
        caravan::after(graphNodeB));
    caravan::syncWait(
        caravan::graph(std::move(graphNodeA), std::move(graphNodeB), std::move(graphNodeC), std::move(graphNodeD)));
    assert(graphA == 2u && graphB == 1u && graphC == 2u && graphD == 1u);

    bool independentGraphNodeRan = false;
    bool failedGraphDescendantSubmitted = false;
    auto graphFailure = caravan::node<"failure">(
        caravan::alpaka::submit(queue, [](Queue&) { throw std::runtime_error("graph submission failed"); }));
    auto graphIndependent = caravan::node<"independent">(caravan::alpaka::submit(
        secondQueue,
        [&](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [&] { independentGraphNodeRan = true; }); }));
    auto graphSkipped = caravan::node<"skipped">(
        caravan::alpaka::submit(queue, [&](Queue&) { failedGraphDescendantSubmitted = true; }),
        caravan::after(graphFailure));
    try
    {
        caravan::syncWait(
            caravan::graph(std::move(graphFailure), std::move(graphIndependent), std::move(graphSkipped)));
        assert(false);
    }
    catch(std::runtime_error const&)
    {
    }
    assert(independentGraphNodeRan && !failedGraphDescendantSubmitted);

    bool mixedNativeRan = false;
    bool mixedHostRan = false;
    auto mixedNative = caravan::node<"native">(caravan::alpaka::submit(
        queue,
        [&](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [&] { mixedNativeRan = true; }); }));
    auto mixedHost = caravan::node<"host">(
        caravan::InlineScheduler{}.schedule()
            | caravan::then(
                [&]
                {
                    assert(mixedNativeRan);
                    mixedHostRan = true;
                }),
        caravan::after(mixedNative));
    caravan::syncWait(caravan::graph(std::move(mixedNative), std::move(mixedHost)));
    assert(mixedNativeRan && mixedHostRan);

    // whenAll and sequence remain native across independent queues and join before the final copy.
    caravan::syncWait(
        caravan::whenAll(
            caravan::alpaka::kernel<Acc>(queue, workDiv, Increment{}, alpaka::getPtrNative(deviceValue)),
            caravan::alpaka::kernel<Acc>(secondQueue, workDiv, Increment{}, alpaka::getPtrNative(crossDevice)))
        | caravan::alpaka::sequence(
            caravan::alpaka::kernel<
                Acc>(queue, workDiv, Add{}, alpaka::getPtrNative(deviceValue), alpaka::getPtrNative(crossDevice)))
        | caravan::alpaka::sequence(caravan::alpaka::copy(queue, hostValue, deviceValue, one)));
    assert(hostValue[0] == 117); // (42 + 1) + (73 + 1)

    bool callbackContextObserved = false;
    scope
        .spawn(
            caravan::alpaka::submit(queue, [](Queue&) {})
            | caravan::then([&] { callbackContextObserved = caravan::alpaka::isCompletionCallback(); }))
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
        caravan::alpaka::submit(queue, [](Queue&) {})
        | caravan::letValue(
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
        caravan::alpaka::fill(queue, alpaka::allocBuf<int, Idx>(device, one), 0u)
        | caravan::continuesOn(stoppedScheduler));
    expectFailed(failedTransfer);

    scope.join().wait();
    assert(callbacks == 8u);
}
