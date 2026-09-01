/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <array>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include <caravan/core.hpp>

#if defined(__unix__)
#    include <sys/wait.h>
#    include <unistd.h>
#endif

namespace
{
    struct InlineExecutor
    {
        template<typename T_Function>
        void post(T_Function&& function)
        {
            std::forward<T_Function>(function)();
        }
    };

    struct EventReceiver
    {
        void set_value() noexcept
        {
            *value = true;
        }

        void set_error(std::exception_ptr error) noexcept
        {
            *failure = std::move(error);
        }

        void set_stopped() noexcept
        {
            *stopped = true;
        }

        bool* value;
        std::exception_ptr* failure;
        bool* stopped;
    };

    template<typename T>
    class AsyncValueSender
    {
    public:
        using completion_signatures = caravan::CompletionSignatures<
            caravan::ValueSignature<T>,
            caravan::ErrorSignature<std::exception_ptr>,
            caravan::StoppedSignature>;

        AsyncValueSender(caravan::Event ready, T value) : m_ready(std::move(ready)), m_value(std::move(value))
        {
        }

        template<typename T_Receiver>
        class Operation
        {
            struct Receiver
            {
                void set_value() noexcept
                {
                    owner->m_receiver.set_value(std::move(owner->m_value));
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->m_receiver.set_error(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->m_receiver.set_stopped();
                }

                Operation* owner;
            };

        public:
            Operation(caravan::Event ready, T value, T_Receiver receiver)
                : m_value(std::move(value))
                , m_receiver(std::move(receiver))
                , m_operation(caravan::asSender(std::move(ready)).connect(Receiver{this}))
            {
            }

            Operation(Operation const&) = delete;
            Operation& operator=(Operation const&) = delete;
            Operation(Operation&&) = delete;
            Operation& operator=(Operation&&) = delete;

            void start() & noexcept
            {
                m_operation.start();
            }

        private:
            T m_value;
            T_Receiver m_receiver;
            caravan::EventOperation<Receiver> m_operation;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{
                std::move(m_ready),
                std::move(m_value),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        caravan::Event m_ready;
        T m_value;
    };

    struct RecursionTrackingExecutor
    {
        template<typename T_Function>
        void post(T_Function&& function)
        {
            ++depth;
            if(depth > maxDepth)
                maxDepth = depth;
            std::forward<T_Function>(function)();
            --depth;
        }

        unsigned depth = 0u;
        unsigned maxDepth = 0u;
    };

    void testCompletionAndContinuations()
    {
        InlineExecutor executor;
        caravan::EventSource source;
        unsigned calls = 0u;
        auto first = source.event().then(executor, [&] { ++calls; });
        assert(source.setReady());
        assert(!source.setReady());
        first.wait();

        auto second = source.event().then(executor, [&] { ++calls; });
        second.wait();
        assert(calls == 2u);
    }

    void testNoRecursiveInlineChains()
    {
        RecursionTrackingExecutor executor;
        caravan::EventSource source;
        auto tail = source.event();
        for(unsigned i = 0u; i < 1000u; ++i)
            tail = tail.then(executor, [] {});
        source.setReady();
        tail.wait();
        assert(executor.maxDepth == 1u);
    }

    void testWhenAllAndFailure()
    {
        caravan::EventSource first;
        caravan::EventSource second;
        std::array events{first.event(), second.event()};
        auto joined = caravan::whenAll(events);
        first.setReady();
        assert(joined.state() == caravan::CompletionState::pending);
        second.setReady();
        joined.wait();

        caravan::EventSource failedInput;
        caravan::EventSource unfinishedInput;
        std::array failingEvents{failedInput.event(), unfinishedInput.event()};
        auto failedJoin = caravan::whenAll(failingEvents);
        failedInput.setFailed(std::make_exception_ptr(std::runtime_error("joined failure")));
        assert(failedJoin.state() == caravan::CompletionState::pending);
        unfinishedInput.setReady();
        try
        {
            failedJoin.wait();
            assert(false);
        }
        catch(std::runtime_error const& error)
        {
            assert(std::string_view{error.what()} == "joined failure");
        }

        caravan::EventSource stoppedInput;
        caravan::EventSource readyInput;
        std::array stoppedEvents{stoppedInput.event(), readyInput.event()};
        auto stoppedJoin = caravan::whenAll(stoppedEvents);
        stoppedInput.setStopped();
        assert(stoppedJoin.state() == caravan::CompletionState::pending);
        readyInput.setReady();
        assert(stoppedJoin.state() == caravan::CompletionState::stopped);

        for(bool stopFirst : {false, true})
        {
            caravan::EventSource failed;
            caravan::EventSource stopped;
            std::array precedenceEvents{failed.event(), stopped.event()};
            auto precedenceJoin = caravan::whenAll(precedenceEvents);
            if(stopFirst)
                stopped.setStopped();
            else
                failed.setFailed(std::make_exception_ptr(std::runtime_error("precedence failure")));
            assert(precedenceJoin.state() == caravan::CompletionState::pending);
            if(stopFirst)
                failed.setFailed(std::make_exception_ptr(std::runtime_error("precedence failure")));
            else
                stopped.setStopped();
            assert(precedenceJoin.state() == caravan::CompletionState::failed);
        }

        for(unsigned i = 0u; i < 100u; ++i)
        {
            caravan::EventSource failed;
            caravan::EventSource stopped;
            std::array concurrentEvents{failed.event(), stopped.event()};
            auto concurrentJoin = caravan::whenAll(concurrentEvents);
            std::thread failer(
                [&] { failed.setFailed(std::make_exception_ptr(std::runtime_error("concurrent failure"))); });
            std::thread stopper([&] { stopped.setStopped(); });
            failer.join();
            stopper.join();
            assert(concurrentJoin.state() == caravan::CompletionState::failed);
        }

        InlineExecutor executor;
        caravan::EventSource failed;
        bool called = false;
        auto successor = failed.event().then(executor, [&] { called = true; });
        failed.setFailed(std::make_exception_ptr(std::runtime_error("expected")));
        try
        {
            successor.wait();
            assert(false);
        }
        catch(std::runtime_error const& error)
        {
            assert(std::string_view{error.what()} == "expected");
        }
        assert(!called);

        bool failureObserved = false;
        auto observation = failed.event().continueWith(
            executor,
            [&](caravan::Event predecessor)
            { failureObserved = predecessor.state() == caravan::CompletionState::failed; });
        observation.wait();
        assert(failureObserved);
    }

    void testFuture()
    {
        InlineExecutor executor;
        caravan::Promise<int> promise;
        auto doubled = promise.future().then(executor, [](int value) { return value * 2; });
        promise.setValue(21);
        assert(doubled.result() == 42);
    }

    void testEventSenderBridge()
    {
        caravan::EventSource source;
        bool value = false;
        bool stopped = false;
        std::exception_ptr failure;
        auto operation = caravan::asSender(source.event()).connect(EventReceiver{&value, &failure, &stopped});

        source.setReady();
        assert(!value);
        operation.start();
        assert(value && !failure && !stopped);

        caravan::EventSource stoppedSource;
        auto stoppedOperation
            = caravan::asSender(stoppedSource.event()).connect(EventReceiver{&value, &failure, &stopped});
        stoppedOperation.start();
        stoppedSource.setStopped();
        assert(stopped);
    }

    void testSyncWait()
    {
        caravan::syncWait(caravan::asSender(caravan::readyEvent()));

        caravan::EventSource failed;
        failed.setFailed(std::make_exception_ptr(std::runtime_error("sync wait failure")));
        try
        {
            caravan::syncWait(caravan::asSender(failed.event()));
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }
    }

    void testLetValue()
    {
        caravan::EventSource predecessor;
        caravan::EventSource successor;
        bool factoryCalled = false;
        auto chain = caravan::letValue(
            caravan::asSender(predecessor.event()),
            [&]
            {
                factoryCalled = true;
                return caravan::asSender(successor.event());
            });
        caravan::AsyncScope scope;
        auto completion = scope.spawn(std::move(chain));
        auto joined = scope.join();

        assert(!factoryCalled);
        predecessor.setReady();
        assert(factoryCalled);
        assert(completion.state() == caravan::CompletionState::pending);
        successor.setReady();
        completion.wait();
        joined.wait();

        caravan::EventSource eagerPredecessor;
        caravan::AsyncScope eagerScope;
        auto eagerCompletion = eagerScope.spawn(
            caravan::letValue(
                caravan::asSender(eagerPredecessor.event()),
                [] { return caravan::asSender(caravan::readyEvent()); }));
        auto eagerJoin = eagerScope.join();
        eagerPredecessor.setReady();
        eagerCompletion.wait();
        eagerJoin.wait();
    }

    void testTypedSenderVocabulary()
    {
        static_assert(caravan::Sender<caravan::EventSender>);
        static_assert(caravan::SenderTo<caravan::EventSender, EventReceiver>);
        static_assert(caravan::Sender<AsyncValueSender<int>>);

        caravan::EventSource thenReady;
        auto doubled
            = caravan::then(AsyncValueSender<int>{thenReady.event(), 21}, [](int value) { return value * 2; });
        static_assert(std::is_same_v<
                      caravan::CompletionSignaturesOf<decltype(doubled)>,
                      caravan::CompletionSignatures<
                          caravan::ValueSignature<int>,
                          caravan::ErrorSignature<std::exception_ptr>,
                          caravan::StoppedSignature>>);
        caravan::AsyncScope thenScope;
        auto doubledResult = thenScope.spawnFuture<int>(std::move(doubled));
        thenReady.setReady();
        assert(doubledResult.result() == 42);
        thenScope.join().wait();

        bool voidThenCalled = false;
        caravan::syncWait(caravan::then(caravan::asSender(caravan::readyEvent()), [&] { voidThenCalled = true; }));
        assert(voidThenCalled);

        caravan::EventSource predecessorReady;
        caravan::EventSource successorReady;
        bool factoryCalled = false;
        auto chained = caravan::letValue(
            AsyncValueSender<int>{predecessorReady.event(), 20},
            [&](int value)
            {
                factoryCalled = true;
                return AsyncValueSender<int>{successorReady.event(), value + 22};
            });
        caravan::AsyncScope letScope;
        auto chainedResult = letScope.spawnFuture<int>(std::move(chained));
        assert(!factoryCalled);
        predecessorReady.setReady();
        assert(factoryCalled);
        successorReady.setReady();
        assert(chainedResult.result() == 42);
        letScope.join().wait();

        caravan::EventSource firstReady;
        caravan::EventSource secondReady;
        auto combined = caravan::then(
            caravan::whenAll(
                AsyncValueSender<int>{firstReady.event(), 40},
                AsyncValueSender<std::string>{secondReady.event(), "ok"}),
            [](int value, std::string text) { return value + static_cast<int>(text.size()); });
        caravan::AsyncScope allScope;
        auto combinedResult = allScope.spawnFuture<int>(std::move(combined));
        firstReady.setReady();
        assert(combinedResult.state() == caravan::CompletionState::pending);
        secondReady.setReady();
        assert(combinedResult.result() == 42);
        allScope.join().wait();

        caravan::syncWait(caravan::whenAll());

        caravan::EventSource failedReady;
        caravan::EventSource unfinishedReady;
        bool continuationCalled = false;
        auto failing = caravan::then(
            caravan::whenAll(
                AsyncValueSender<int>{failedReady.event(), 1},
                AsyncValueSender<int>{unfinishedReady.event(), 2}),
            [&](int, int) { continuationCalled = true; });
        caravan::AsyncScope failureScope;
        auto failed = failureScope.spawn(std::move(failing));
        failedReady.setFailed(std::make_exception_ptr(std::runtime_error("typed whenAll failure")));
        assert(failed.state() == caravan::CompletionState::pending);
        unfinishedReady.setReady();
        assert(failed.state() == caravan::CompletionState::failed);
        assert(!continuationCalled);
        failureScope.join().wait();

        caravan::EventSource stoppedReady;
        caravan::EventSource stoppedPeerReady;
        caravan::AsyncScope stoppedScope;
        auto stopped = stoppedScope.spawn(
            caravan::whenAll(
                AsyncValueSender<int>{stoppedReady.event(), 1},
                AsyncValueSender<int>{stoppedPeerReady.event(), 2}));
        stoppedReady.setStopped();
        assert(stopped.state() == caravan::CompletionState::pending);
        stoppedPeerReady.setReady();
        assert(stopped.state() == caravan::CompletionState::stopped);
        stoppedScope.join().wait();
    }

    void testEagerSenderBridgesAndOperationLifetime()
    {
        caravan::AsyncScope scope;
        caravan::EventSource valueReady;
        auto value = scope.spawnFuture<int>(AsyncValueSender<int>{valueReady.event(), 42});
        assert(value.state() == caravan::CompletionState::pending);
        valueReady.setReady();
        assert(value.result() == 42);

        caravan::EventSource failedReady;
        auto failed = scope.spawnFuture<int>(AsyncValueSender<int>{failedReady.event(), 0});
        failedReady.setFailed(std::make_exception_ptr(std::runtime_error("typed bridge failure")));
        try
        {
            static_cast<void>(failed.result());
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }

        caravan::EventSource stoppedReady;
        auto stopped = scope.spawnFuture<int>(AsyncValueSender<int>{stoppedReady.event(), 0});
        stoppedReady.setStopped();
        try
        {
            static_cast<void>(stopped.result());
            assert(false);
        }
        catch(caravan::StoppedError const&)
        {
        }

        caravan::EventSource predecessor;
        caravan::EventSource successor;
        auto retained = std::make_shared<int>(7);
        std::weak_ptr<int> lifetime = retained;
        auto completion = scope.spawn(
            caravan::letValue(
                caravan::asSender(predecessor.event()),
                [retained, ready = successor.event()] { return caravan::asSender(ready); }));
        retained.reset();
        assert(!lifetime.expired());
        predecessor.setReady();
        assert(!lifetime.expired());
        successor.setReady();
        completion.wait();
        assert(lifetime.expired());
        scope.join().wait();
    }

    void testContinuesOnRunLoop()
    {
        caravan::RunLoop runLoop;
        auto scheduler = runLoop.scheduler();
        static_assert(std::is_trivially_copyable_v<caravan::RunLoopScheduler>);
        caravan::AsyncScope scope;
        caravan::EventSource source;
        auto transferred = scope.spawn(caravan::continuesOn(caravan::asSender(source.event()), scheduler));
        auto joined = scope.join();

        source.setReady();
        assert(transferred.state() == caravan::CompletionState::pending);
        assert(joined.state() == caravan::CompletionState::pending);
        runLoop.finish();
        runLoop.run();
        transferred.wait();
        joined.wait();
        try
        {
            scheduler.post([] {});
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }

        caravan::RunLoop polledLoop;
        caravan::AsyncScope polledScope;
        caravan::EventSource polledSource;
        auto polled
            = polledScope.spawn(caravan::continuesOn(caravan::asSender(polledSource.event()), polledLoop.scheduler()));
        auto polledJoin = polledScope.join();
        polledSource.setReady();
        assert(polled.state() == caravan::CompletionState::pending);
        polledLoop.runReady();
        polled.wait();
        polledJoin.wait();
    }

    void testAsyncScope()
    {
        caravan::AsyncScope scope;
        assert(scope.status() == caravan::AsyncScopeStatus::open);
        caravan::EventSource readySource;
        caravan::EventSource failedSource;
        caravan::EventSource stoppedSource;
        auto ready = scope.spawn(caravan::asSender(readySource.event()));
        auto failed = scope.spawn(caravan::asSender(failedSource.event()));
        auto stopped = scope.spawn(caravan::asSender(stoppedSource.event()));
        auto joined = scope.join();

        assert(scope.status() == caravan::AsyncScopeStatus::joining);
        assert(joined.state() == caravan::CompletionState::pending);
        readySource.setReady();
        failedSource.setFailed(std::make_exception_ptr(std::runtime_error("scope failure")));
        stoppedSource.setStopped();
        joined.wait();
        assert(scope.status() == caravan::AsyncScopeStatus::joined);
        assert(ready.isReady());
        assert(failed.state() == caravan::CompletionState::failed && failed.error());
        assert(stopped.state() == caravan::CompletionState::stopped);

        try
        {
            scope.spawn(caravan::asSender(caravan::readyEvent()));
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }
    }

    void testPendingScopeDestructionDiagnosed()
    {
#if defined(__unix__)
        auto const child = fork();
        assert(child >= 0);
        if(child == 0)
        {
            std::set_terminate([] { std::_Exit(42); });
            {
                caravan::RunLoop loop;
                caravan::AsyncScope scope;
                static_cast<void>(
                    scope.spawn(caravan::continuesOn(caravan::asSender(caravan::readyEvent()), loop.scheduler())));
            }
            std::_Exit(0);
        }

        int status = 0;
        assert(waitpid(child, &status, 0) == child);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 42);
#endif
    }

    void testExactlyOnceCompletion()
    {
        caravan::EventSource source;
        std::atomic<unsigned> winners = 0u;
        std::vector<std::thread> threads;
        for(unsigned i = 0u; i < 8u; ++i)
            threads.emplace_back([&] { winners.fetch_add(source.setReady()); });
        for(auto& thread : threads)
            thread.join();
        assert(winners == 1u);
    }

    void testRegistrationRace()
    {
        InlineExecutor executor;
        caravan::EventSource source;
        constexpr unsigned threadCount = 8u;
        constexpr unsigned continuationsPerThread = 100u;
        std::atomic<unsigned> calls = 0u;
        std::vector<caravan::Event> completions;
        std::mutex completionsMutex;
        std::vector<std::thread> threads;

        for(unsigned thread = 0u; thread < threadCount; ++thread)
        {
            threads.emplace_back(
                [&]
                {
                    std::vector<caravan::Event> local;
                    for(unsigned i = 0u; i < continuationsPerThread; ++i)
                        local.emplace_back(source.event().then(executor, [&] { calls.fetch_add(1u); }));
                    std::lock_guard lock(completionsMutex);
                    completions.insert(completions.end(), local.begin(), local.end());
                });
        }
        std::thread completer([&] { source.setReady(); });
        for(auto& thread : threads)
            thread.join();
        completer.join();
        caravan::whenAll(completions).wait();
        assert(calls == threadCount * continuationsPerThread);
    }

    void testExecutorWaitGuard()
    {
        InlineExecutor executor;
        caravan::EventSource start;
        caravan::EventSource pending;
        auto blocked = start.event().then(executor, [&] { pending.event().wait(); });
        start.setReady();
        try
        {
            blocked.wait();
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }
    }
} // namespace

int main()
{
    assert(caravan::readyEvent().isReady());
    testCompletionAndContinuations();
    testNoRecursiveInlineChains();
    testWhenAllAndFailure();
    testFuture();
    testEventSenderBridge();
    testSyncWait();
    testLetValue();
    testTypedSenderVocabulary();
    testEagerSenderBridgesAndOperationLifetime();
    testContinuesOnRunLoop();
    testAsyncScope();
    testPendingScopeDestructionDiagnosed();
    testExactlyOnceCompletion();
    testRegistrationRace();
    testExecutorWaitGuard();
}
