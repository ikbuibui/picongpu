/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <array>
#include <atomic>
#include <cassert>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include <caravan/core.hpp>
#include <caravan/sender.hpp>

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
        caravan::AsyncScope scope;
        caravan::EventSource source;
        auto transferred = scope.spawn(caravan::continuesOn(caravan::asSender(source.event()), runLoop));
        auto joined = scope.join();

        source.setReady();
        assert(transferred.state() == caravan::CompletionState::pending);
        assert(joined.state() == caravan::CompletionState::pending);
        runLoop.finish();
        runLoop.run();
        transferred.wait();
        joined.wait();

        caravan::RunLoop polledLoop;
        caravan::AsyncScope polledScope;
        caravan::EventSource polledSource;
        auto polled = polledScope.spawn(caravan::continuesOn(caravan::asSender(polledSource.event()), polledLoop));
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
        caravan::EventSource readySource;
        caravan::EventSource failedSource;
        caravan::EventSource stoppedSource;
        auto ready = scope.spawn(caravan::asSender(readySource.event()));
        auto failed = scope.spawn(caravan::asSender(failedSource.event()));
        auto stopped = scope.spawn(caravan::asSender(stoppedSource.event()));
        auto joined = scope.join();

        assert(joined.state() == caravan::CompletionState::pending);
        readySource.setReady();
        failedSource.setFailed(std::make_exception_ptr(std::runtime_error("scope failure")));
        stoppedSource.setStopped();
        joined.wait();
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
    testEagerSenderBridgesAndOperationLifetime();
    testContinuesOnRunLoop();
    testAsyncScope();
    testExactlyOnceCompletion();
    testRegistrationRace();
    testExecutorWaitGuard();
}
