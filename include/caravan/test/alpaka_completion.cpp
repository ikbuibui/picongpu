/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

namespace
{
    thread_local unsigned allocationFailures = 0u;
    std::weak_ptr<int> fatalOwner;
    std::atomic<bool> fatalTaskFinished = false;

    void expectFailed(caravan::Event const& event)
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
    }
} // namespace

// Inject only on the submitting thread, after a native task has borrowed retained storage.
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
void* operator new(std::size_t bytes)
{
    if(allocationFailures != 0u)
    {
        --allocationFailures;
        throw std::bad_alloc{};
    }
    if(auto* memory = std::malloc(bytes == 0u ? 1u : bytes))
        return memory;
    throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}
#endif

int main(int argc, char** argv)
{
    // These regressions exercise CPU queues even in CUDA/HIP builds; no accelerator device is required.
    using Queue = alpaka::QueueCpuNonBlocking;
    auto const device = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
    Queue queue{device};
    Queue secondQueue{device};
    Queue blockerQueue{device};
    caravan::AsyncScope scope;

    if(argc > 1 && std::string_view(argv[1]) == "--fatal-fence")
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto storage = std::make_shared<int>(42);
        fatalOwner = storage;
        std::set_terminate(
            [] { std::_Exit(!fatalOwner.expired() && !fatalTaskFinished.load() ? EXIT_SUCCESS : EXIT_FAILURE); });
        scope.spawn(
            caravan::alpaka::submit(
                queue,
                [storage = std::move(storage), gate](Queue& nativeQueue)
                {
                    alpaka::enqueue(
                        nativeQueue,
                        [raw = storage.get(), gate]
                        {
                            gate.wait();
                            assert(*raw == 42);
                            fatalTaskFinished = true;
                        });
                    allocationFailures = std::numeric_limits<unsigned>::max();
                }));
        // Expected termination must occur in cleanup, before receiver delivery or scope destruction.
        std::_Exit(EXIT_FAILURE);
    }

    // Native fork/join submits the join before host completion, without serializing branches.
    // The fork follows a seed on queue A; branch B must start even while branch A is blocked.
    {
        caravan::alpaka::Scheduler scheduler{queue};
        static_assert(
            std::is_same_v<decltype(caravan::getDomain(scheduler)), caravan::alpaka::SubmissionDomain<Queue>>);
        std::promise<void> release, branchStarted;
        auto gate = release.get_future().share();
        auto started = branchStarted.get_future();
        int seed = 0, left = 0, right = 0;
        bool joinSubmitted = false;
        std::atomic<bool> hostCompleted = false;
        auto work = scheduler.submit([&](Queue& q) { alpaka::enqueue(q, [&] { seed = 42; }); })
                    | caravan::alpaka::sequence(
                        caravan::whenAll(
                            scheduler.submit(
                                [&](Queue& q)
                                {
                                    alpaka::enqueue(
                                        q,
                                        [&]
                                        {
                                            gate.wait();
                                            left = seed;
                                        });
                                }),
                            caravan::whenAll(
                                caravan::alpaka::submit(
                                    secondQueue,
                                    [&](Queue& q)
                                    {
                                        alpaka::enqueue(
                                            q,
                                            [&]
                                            {
                                                branchStarted.set_value();
                                                gate.wait();
                                                right = seed;
                                            });
                                    }),
                                caravan::alpaka::submit(blockerQueue, [](Queue&) {}))))
                    | caravan::alpaka::sequence(scheduler.submit(
                        [&, owner = std::make_unique<int>(42)](Queue& q)
                        {
                            joinSubmitted = true;
                            alpaka::enqueue(q, [&, raw = owner.get()] { assert(left == *raw && right == *raw); });
                        }))
                    | caravan::then(
                        [&]
                        {
                            assert(left == 42 && right == 42);
                            hostCompleted = true;
                        });
        auto done = scope.spawn(caravan::startsOn(scheduler, std::move(work)));
        assert(joinSubmitted);
        assert(!hostCompleted);
        assert(done.state() == caravan::CompletionState::pending);
        started.get();
        release.set_value();
        done.wait();
        assert(hostCompleted);
    }

    // A mixed join uses letValue to start the next submission after host completion.
    {
        caravan::EventSource release;
        std::atomic<bool> submitted = false;
        auto done = scope.spawn(
            caravan::whenAll(caravan::alpaka::submit(queue, [](Queue&) {}), caravan::asSender(release.event()))
            | caravan::letValue([&]
                                { return caravan::alpaka::submit(secondQueue, [&](Queue&) { submitted = true; }); }));
        assert(!submitted);
        release.setReady();
        done.wait();
        assert(submitted);
    }

    // Neither an ordinary host callback nor explicit placement can be fused away.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        std::atomic<bool> submitted = false;
        auto done = scope.spawn(
            caravan::alpaka::submit(queue, [gate](Queue& q) { alpaka::enqueue(q, [gate] { gate.wait(); }); })
            | caravan::then([] { assert(caravan::isExecutorThread()); })
            | caravan::letValue([&]
                                { return caravan::alpaka::submit(secondQueue, [&](Queue&) { submitted = true; }); }));
        assert(!submitted);
        release.set_value();
        done.wait();
        assert(submitted);

        submitted = false;
        caravan::RunLoop loop;
        auto placed = scope.spawn(
            caravan::alpaka::submit(queue, [](Queue&) {}) | caravan::continuesOn(loop.scheduler())
            | caravan::letValue([&]
                                { return caravan::alpaka::submit(secondQueue, [&](Queue&) { submitted = true; }); }));
        assert(!submitted);
        while(placed.state() == caravan::CompletionState::pending)
        {
            loop.runReady();
            std::this_thread::yield();
        }
        placed.wait();
        assert(submitted);
    }

    // Failed submission skips descendants, starts independent branches, and retains partially submitted work.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto owner = std::make_shared<int>(42);
        std::weak_ptr<int> observer = owner;
        bool siblingSubmitted = false, descendantSubmitted = false, joinSubmitted = false;
        caravan::AsyncScope failureScope;
        auto failed = failureScope.spawn(
            caravan::whenAll(
                caravan::alpaka::submit(
                    queue,
                    [owner = std::move(owner), gate](Queue& q)
                    {
                        alpaka::enqueue(
                            q,
                            [raw = owner.get(), gate]
                            {
                                gate.wait();
                                assert(*raw == 42);
                            });
                        throw std::runtime_error("partial branch");
                    })
                    | caravan::alpaka::sequence(
                        caravan::alpaka::submit(queue, [&](Queue&) { descendantSubmitted = true; })),
                caravan::alpaka::submit(secondQueue, [&](Queue&) { siblingSubmitted = true; }))
            | caravan::alpaka::sequence(caravan::alpaka::submit(blockerQueue, [&](Queue&) { joinSubmitted = true; })));
        assert(siblingSubmitted && !descendantSubmitted && !joinSubmitted);
        assert(failed.state() == caravan::CompletionState::pending);
        assert(!observer.expired());
        release.set_value();
        expectFailed(failed);
        failureScope.join().wait();
        assert(observer.expired());
    }

    // Asynchronous branch errors are reported at quiescence, not used to retract an already submitted join.
    {
        bool joinSubmitted = false;
        auto failed = scope.spawn(
            caravan::whenAll(
                caravan::alpaka::submit(
                    queue,
                    [](Queue& q) { alpaka::enqueue(q, [] { throw std::runtime_error("branch execution"); }); }),
                caravan::alpaka::submit(secondQueue, [](Queue&) {}))
            | caravan::alpaka::sequence(caravan::alpaka::submit(blockerQueue, [&](Queue&) { joinSubmitted = true; })));
        assert(joinSubmitted);
        expectFailed(failed);
    }

    // A pending queue must not hold up a ready queue whose continuation releases it.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto pending = scope.spawn(
            caravan::alpaka::submit(
                queue,
                [gate](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [gate] { gate.wait(); }); }));
        scope.spawn(caravan::alpaka::submit(secondQueue, [](Queue&) {}) | caravan::then([&] { release.set_value(); }))
            .wait();
        pending.wait();
    }

    // A late observer must not include a subsequent operation's error in an earlier operation's fence.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto blocker = scope.spawn(
            caravan::alpaka::submit(
                blockerQueue,
                [gate](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [gate] { gate.wait(); }); }));
        auto good = scope.spawn(caravan::alpaka::submit(queue, [](Queue&) {}));
        auto bad = scope.spawn(
            caravan::alpaka::submit(
                queue,
                [](Queue& nativeQueue)
                { alpaka::enqueue(nativeQueue, [] { throw std::runtime_error("later operation"); }); }));
        release.set_value();
        good.wait();
        expectFailed(bad);
        blocker.wait();
    }

    // Work appended after a sender may depend on that sender's completion without moving its fence.
    {
        std::promise<void> release;
        auto gate = release.get_future().share();
        auto blocker = scope.spawn(
            caravan::alpaka::submit(
                blockerQueue,
                [gate](Queue& nativeQueue) { alpaka::enqueue(nativeQueue, [gate] { gate.wait(); }); }));
        auto predecessor = scope.spawn(caravan::alpaka::submit(queue, [](Queue&) {}));
        alpaka::enqueue(queue, [predecessor] { predecessor.wait(); });
        release.set_value();
        predecessor.wait();
        blocker.wait();
        alpaka::wait(queue);
    }

    // Cross-queue error snapshots survive dependency waits, including a return to an earlier queue.
    auto crossQueueFailure = scope.spawn(
        caravan::alpaka::sequence(
            caravan::alpaka::sequence(
                caravan::alpaka::submit(
                    queue,
                    [](Queue& nativeQueue)
                    { alpaka::enqueue(nativeQueue, [] { throw std::runtime_error("first queue"); }); }),
                caravan::alpaka::submit(secondQueue, [](Queue&) {})),
            caravan::alpaka::submit(queue, [](Queue&) {})));
    expectFailed(crossQueueFailure);
    scope.spawn(caravan::alpaka::submit(queue, [](Queue&) {})).wait();

    // Cleanup fences the active same-queue run even when its final stages were never submitted.
    bool skippedStageRan = false;
    auto partialFailure = scope.spawn(
        caravan::alpaka::sequence(
            caravan::alpaka::submit(queue, [](Queue&) {}),
            caravan::alpaka::sequence(
                caravan::alpaka::submit(secondQueue, [](Queue&) { throw std::runtime_error("partial chain"); }),
                caravan::alpaka::submit(secondQueue, [&](Queue&) { skippedStageRan = true; }))));
    expectFailed(partialFailure);
    assert(!skippedStageRan);

    // Blocking CPU queues exercise the native-event fence implementation too.
    alpaka::QueueCpuBlocking blockingQueue{device};
    alpaka::QueueCpuBlocking secondBlockingQueue{device};
    scope
        .spawn(
            caravan::alpaka::sequence(
                caravan::alpaka::submit(blockingQueue, [](auto&) {}),
                caravan::alpaka::submit(secondBlockingQueue, [](auto&) {})))
        .wait();

    // Receiver delivery uses the same blocking guards as the other Caravan progress authorities.
    auto nestedWait = scope.spawn(
        caravan::alpaka::submit(queue, [](Queue&) {})
        | caravan::then(
            [&]
            {
                assert(caravan::isExecutorThread());
                caravan::syncWait(caravan::alpaka::submit(secondQueue, [](Queue&) {}));
            }));
    expectFailed(nestedWait);

    // Failure to allocate the first fence must retain captures until a recovery fence proves completion.
    std::promise<void> release;
    auto gate = release.get_future().share();
    auto storage = std::make_shared<int>(42);
    std::weak_ptr<int> observer = storage;
    auto failedFence = scope.spawn(
        caravan::alpaka::submit(
            queue,
            [storage = std::move(storage), gate](Queue& nativeQueue)
            {
                alpaka::enqueue(
                    nativeQueue,
                    [raw = storage.get(), gate]
                    {
                        gate.wait();
                        assert(*raw == 42);
                    });
                allocationFailures = 1u;
            }));
    scope.spawn(caravan::alpaka::submit(secondQueue, [](Queue&) {})).wait();
    assert(failedFence.state() == caravan::CompletionState::pending);
    assert(!observer.expired());
    release.set_value();
    expectFailed(failedFence);

    scope.join().wait();
    assert(observer.expired());
}
