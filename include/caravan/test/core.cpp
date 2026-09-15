/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <array>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <functional>
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
    struct UnsupportedMultiValueSender
    {
        using completion_signatures = caravan::CompletionSignatures<
            caravan::ValueSignature<int>,
            caravan::ValueSignature<double>,
            caravan::ErrorSignature<std::exception_ptr>>;
    };

    struct UnsupportedErrorSender
    {
        using completion_signatures
            = caravan::CompletionSignatures<caravan::ValueSignature<>, caravan::ErrorSignature<int>>;
    };

    struct UnsupportedReferenceSender
    {
        using completion_signatures = caravan::
            CompletionSignatures<caravan::ValueSignature<int&>, caravan::ErrorSignature<std::exception_ptr>>;
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

        bool* value;
        std::exception_ptr* failure;
    };

    template<typename T>
    class AsyncValueSender
    {
    public:
        using completion_signatures
            = caravan::CompletionSignatures<caravan::ValueSignature<T>, caravan::ErrorSignature<std::exception_ptr>>;

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

    struct StartTrackingSender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

        template<typename T_Receiver>
        struct Operation
        {
            void start() & noexcept
            {
                *started = true;
                receiver.set_value();
            }

            bool* started;
            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{started, std::forward<T_Receiver>(receiver)};
        }

        bool* started;
    };

    struct ScopeOperationTrackingSender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

        template<typename T_Receiver>
        struct Operation
        {
            ~Operation()
            {
                *destroyed = true;
            }

            void start() & noexcept
            {
                receiver.set_value();
            }

            bool* destroyed;
            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{destroyed, std::forward<T_Receiver>(receiver)};
        }

        bool* destroyed;
    };

    struct ThrowingConnectSender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

        template<typename T_Receiver>
        struct Operation
        {
            void start() & noexcept
            {
                receiver.set_value();
            }

            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) && -> Operation<std::decay_t<T_Receiver>>
        {
            static_cast<void>(receiver);
            throw std::runtime_error("connect failed");
        }
    };

    struct EnvironmentReceiver
    {
        void set_value(int value) noexcept
        {
            *output = value;
        }

        void set_error(std::exception_ptr) noexcept
        {
            assert(false);
        }

        int get_env() const noexcept
        {
            return 10;
        }

        int* output;
    };

    struct EnvironmentSender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<int>>;

        template<typename T_Receiver>
        struct Operation
        {
            void start() & noexcept
            {
                receiver.set_value(value);
            }

            int value;
            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            *observed += receiver.get_env();
            return Operation<std::decay_t<T_Receiver>>{value, std::forward<T_Receiver>(receiver)};
        }

        int value;
        int* observed;
    };

    struct GetMarker
    {
        template<typename T_Environment>
        auto operator()(T_Environment const& environment) const noexcept -> decltype(environment.query(*this))
        {
            return environment.query(*this);
        }
    };

    struct QueryEnvironment
    {
        QueryEnvironment() = default;
        QueryEnvironment(QueryEnvironment const&) = delete;

        int query(GetMarker) const noexcept
        {
            return 42;
        }
    };

    struct QueryReceiver : EventReceiver
    {
        QueryEnvironment const& get_env() const noexcept
        {
            return *environment;
        }

        QueryEnvironment const* environment;
    };

    struct TaggedScheduler
    {
        auto schedule() const
        {
            return caravan::InlineScheduler{}.schedule();
        }

        int id;
    };

    struct QuerySender
    {
        using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

        template<typename T_Receiver>
        struct Operation
        {
            void start() & noexcept
            {
                assert(caravan::getScheduler(receiver.get_env()).id == expected);
                assert(GetMarker{}(receiver.get_env()) == 42);
                ++*observations;
                receiver.set_value();
            }

            int expected;
            int* observations;
            T_Receiver receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver receiver) &&
        {
            assert(caravan::getScheduler(receiver.get_env()).id == expected);
            assert(GetMarker{}(receiver.get_env()) == 42);
            ++*observations;
            return Operation<T_Receiver>{expected, observations, std::move(receiver)};
        }

        int expected;
        int* observations;
    };

    struct EventScheduler
    {
        auto schedule() const
        {
            return caravan::asSender(ready);
        }

        caravan::Event ready;
    };

    template<typename T_Scheduler>
    struct SchedulerReceiver : EventReceiver
    {
        struct Environment
        {
            T_Scheduler query(caravan::GetScheduler) const
            {
                return scheduler;
            }

            T_Scheduler scheduler;
        };

        Environment get_env() const
        {
            return {scheduler};
        }

        T_Scheduler scheduler;
    };

    struct ThrowOnMove
    {
        ThrowOnMove() = default;

        ThrowOnMove(ThrowOnMove&&)
        {
            throw std::runtime_error("value storage");
        }
    };

    struct RecursionTrackingScheduler
    {
        struct ScheduleSender
        {
            using completion_signatures = caravan::detail::DefaultCompletionSignatures<caravan::ValueSignature<>>;

            template<typename T_Receiver>
            struct Operation
            {
                void start() & noexcept
                {
                    auto* currentDepth = depth;
                    ++*currentDepth;
                    if(*currentDepth > *maxDepth)
                        *maxDepth = *currentDepth;
                    receiver.set_value();
                    --*currentDepth;
                }

                unsigned* depth;
                unsigned* maxDepth;
                T_Receiver receiver;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver receiver) &&
            {
                return Operation<T_Receiver>{depth, maxDepth, std::move(receiver)};
            }

            unsigned* depth;
            unsigned* maxDepth;
        };

        auto schedule() const
        {
            return ScheduleSender{depth, maxDepth};
        }

        unsigned* depth;
        unsigned* maxDepth;
    };

    void testInlineScheduler()
    {
        caravan::InlineScheduler scheduler;
        bool completed = false;
        std::exception_ptr failure;
        auto operation = scheduler.schedule().connect(EventReceiver{&completed, &failure});
        assert(!completed);
        operation.start();
        assert(completed && !failure);
    }

    void testCompletionAndContinuations()
    {
        caravan::InlineScheduler executor;
        caravan::EventSource source;
        unsigned calls = 0u;
        caravan::AsyncScope scope;
        auto first = scope.spawn(
            caravan::asSender(source.event()) | caravan::continuesOn(executor) | caravan::then([&] { ++calls; }));
        auto const firstCompletion = source.setReady();
        auto const duplicateCompletion = source.setReady();
        assert(firstCompletion);
        assert(!duplicateCompletion);
        first.wait();

        auto second = scope.spawn(
            caravan::asSender(source.event()) | caravan::continuesOn(executor) | caravan::then([&] { ++calls; }));
        second.wait();
        assert(calls == 2u);
        scope.join().wait();
    }

    void testRunReadyFairness()
    {
        caravan::RunLoop loop;
        auto scheduler = loop.scheduler();
        unsigned runs = 0u;
        caravan::AsyncScope scope;
        std::function<void()> repost;
        repost = [&]
        {
            ++runs;
            if(runs < 3u)
                scope.spawn(scheduler.schedule() | caravan::then(repost));
        };
        scope.spawn(scheduler.schedule() | caravan::then(repost));

        loop.runReady();
        assert(runs == 1u);
        loop.runReady();
        assert(runs == 2u);
        loop.runReady();
        assert(runs == 3u);

        bool scheduled = false;
        auto completion = scope.spawn(scheduler.schedule() | caravan::then([&] { scheduled = true; }));
        assert(!scheduled && completion.state() == caravan::CompletionState::pending);
        loop.runReady();
        completion.wait();
        assert(scheduled);
        scope.join().wait();
    }

    void testSchedulerHandleLifetime()
    {
        caravan::RunLoop loop;
        caravan::EventSource source;
        caravan::Promise<int> promise;
        caravan::Event continued;
        caravan::Event observed;
        caravan::Future<int> mapped;
        bool ran = false;
        bool observationRan = false;
        caravan::AsyncScope scope;
        {
            auto scheduler = loop.scheduler();
            continued = scope.spawn(
                caravan::asSender(source.event()) | caravan::continuesOn(scheduler)
                | caravan::then([&] { ran = true; }));
            observed = source.event().continueWith(scheduler, [&](caravan::Event) { observationRan = true; });
            mapped = scope.spawnFuture<int>(
                caravan::consumeAsSender(promise.future()) | caravan::continuesOn(scheduler)
                | caravan::then([](int value) { return value * 2; }));
        }

        source.setReady();
        promise.setValue(21);
        loop.runReady();
        continued.wait();
        observed.wait();
        assert(mapped.result() == 42);
        assert(ran && observationRan);
        scope.join().wait();
    }

    void testNoRecursiveInlineChains()
    {
        unsigned depth = 0u;
        unsigned maxDepth = 0u;
        RecursionTrackingScheduler scheduler{&depth, &maxDepth};
        caravan::EventSource source;
        caravan::AsyncScope scope;
        auto tail = source.event();
        for(unsigned i = 0u; i < 1000u; ++i)
            tail = scope.spawn(caravan::asSender(tail) | caravan::continuesOn(scheduler));
        source.setReady();
        tail.wait();
        assert(maxDepth == 1u);
        scope.join().wait();
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


        caravan::InlineScheduler executor;
        caravan::EventSource failed;
        bool called = false;
        caravan::AsyncScope scope;
        auto successor = scope.spawn(
            caravan::asSender(failed.event())
            | caravan::letValue([&] { return executor.schedule() | caravan::then([&] { called = true; }); }));
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
        scope.join().wait();
    }

    void testFuture()
    {
        caravan::InlineScheduler executor;
        caravan::Promise<std::unique_ptr<int>> promise;
        auto shared = promise.future();
        caravan::AsyncScope scope;
        auto observe = [&]
        {
            return scope.spawnFuture<int>(
                caravan::asSender(shared.event())
                | caravan::letValue(
                    [shared, executor]
                    { return executor.schedule() | caravan::then([shared] { return *shared.result() * 2; }); }));
        };
        auto doubled = observe();
        auto peer = observe();
        promise.setValue(std::make_unique<int>(21));
        assert(doubled.result() == 42 && peer.result() == 42);
        assert(observe().result() == 42);
        assert(*shared.result() == 21); // Observation did not consume the shared value.
        scope.join().wait();
    }

    void testEventSenderBridge()
    {
        caravan::EventSource source;
        bool value = false;
        std::exception_ptr failure;
        auto operation = caravan::asSender(source.event()).connect(EventReceiver{&value, &failure});

        source.setReady();
        assert(!value);
        operation.start();
        assert(value && !failure);
    }

    void testSyncWait()
    {
        caravan::syncWait(caravan::asSender(caravan::readyEvent()));
        auto moveOnly = caravan::syncWait<std::unique_ptr<int>>(
            AsyncValueSender<std::unique_ptr<int>>{caravan::readyEvent(), std::make_unique<int>(42)});
        assert(*moveOnly == 42);

        bool started = false;
        {
            caravan::ExecutorThreadGuard guard;
            try
            {
                caravan::syncWait(StartTrackingSender{&started});
                assert(false);
            }
            catch(std::logic_error const&)
            {
            }
        }
        assert(!started);

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
        auto chain = caravan::asSender(predecessor.event())
                     | caravan::letValue(
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
            caravan::asSender(eagerPredecessor.event())
            | caravan::letValue([] { return caravan::asSender(caravan::readyEvent()); }));
        auto eagerJoin = eagerScope.join();
        eagerPredecessor.setReady();
        eagerCompletion.wait();
        eagerJoin.wait();

        caravan::EventSource valueReady;
        caravan::EventSource borrowedBySuccessor;
        auto value = std::make_shared<int>(42);
        std::weak_ptr<int> valueLifetime = value;
        caravan::AsyncScope valueScope;
        auto valueCompletion = valueScope.spawn(
            AsyncValueSender<std::shared_ptr<int>>{valueReady.event(), std::move(value)}
            | caravan::letValue(
                [&borrowedBySuccessor](std::shared_ptr<int> const& stored)
                {
                    assert(*stored == 42);
                    return caravan::asSender(borrowedBySuccessor.event());
                }));
        valueReady.setReady();
        assert(!valueLifetime.expired());
        borrowedBySuccessor.setReady();
        valueCompletion.wait();
        assert(valueLifetime.expired());
        valueScope.join().wait();
    }

    void testSequence()
    {
        caravan::EventSource predecessor, successor;
        bool started = false;
        auto work = AsyncValueSender<std::unique_ptr<int>>{predecessor.event(), std::make_unique<int>(1)}
                    | caravan::sequence(
                        caravan::sequence(
                            StartTrackingSender{&started},
                            AsyncValueSender<std::unique_ptr<int>>{successor.event(), std::make_unique<int>(42)}));
        static_assert(std::is_same_v<
                      caravan::CompletionSignaturesOf<decltype(work)>,
                      caravan::CompletionSignaturesOf<AsyncValueSender<std::unique_ptr<int>>>>);
        assert(!started);
        caravan::AsyncScope scope;
        auto result = scope.spawnFuture<std::unique_ptr<int>>(std::move(work));
        assert(!started);
        predecessor.setReady();
        assert(started && result.state() == caravan::CompletionState::pending);
        successor.setReady();
        assert(*std::move(result).takeResult() == 42);

        // Join failure drains every branch and never starts the successor.
        caravan::EventSource failed, pending;
        started = false;
        auto error = std::make_exception_ptr(std::runtime_error("sequence predecessor"));
        auto failure = scope.spawn(
            caravan::whenAll(caravan::asSender(failed.event()), caravan::asSender(pending.event()))
            | caravan::sequence(StartTrackingSender{&started}));
        failed.setFailed(error);
        assert(failure.state() == caravan::CompletionState::pending && !started);
        pending.setReady();
        assert(failure.state() == caravan::CompletionState::failed && failure.error() == error && !started);

        // Connecting the successor is deferred; connection errors use the error channel.
        auto connectFailure
            = scope.spawn(caravan::sequence(caravan::asSender(caravan::readyEvent()), ThrowingConnectSender{}));
        assert(connectFailure.state() == caravan::CompletionState::failed);
        try
        {
            connectFailure.wait();
            assert(false);
        }
        catch(std::runtime_error const& failure)
        {
            assert(std::string_view{failure.what()} == "connect failed");
        }
        scope.join().wait();

        int observations = 0, output = 0;
        auto environmentWork
            = caravan::sequence(EnvironmentSender{1, &observations}, EnvironmentSender{42, &observations});
        auto operation = std::move(environmentWork).connect(EnvironmentReceiver{&output});
        assert(observations == 10 && output == 0);
        operation.start();
        assert(observations == 20 && output == 42);
    }

    void testRepeatUntil()
    {
        // A move-only factory owns iteration state. Completed children must be reclaimed before reuse.
        bool destroyed = true;
        unsigned calls = 0u;
        unsigned depth = 0u;
        unsigned maxDepth = 0u;
        auto sender = caravan::repeatUntil(
            [&, count = std::make_unique<unsigned>(0u)]
            {
                assert(destroyed);
                destroyed = false;
                ++calls;
                return caravan::whenAll(
                           ScopeOperationTrackingSender{&destroyed},
                           RecursionTrackingScheduler{&depth, &maxDepth}.schedule())
                       | caravan::then([value = count.get()] { return ++*value == 100000u; });
            });
        static_assert(caravan::SenderTo<decltype(sender), EventReceiver>);
        assert(calls == 0u);
        caravan::AsyncScope scope;
        auto result = scope.spawn(std::move(sender)); // Terminal completion destroys the loop inline.
        scope.join().wait();
        assert(result.isReady() && calls == 100000u && destroyed && maxDepth == 1u && depth == 0u);

        bool value = false;
        std::exception_ptr error;
        int observations = 0;
        unsigned iterations = 0u;
        QueryEnvironment environment;
        auto work = caravan::startsOn(
            TaggedScheduler{1},
            caravan::repeatUntil(
                [&] { return QuerySender{1, &observations} | caravan::then([&] { return ++iterations == 3u; }); }));
        auto operation = std::move(work).connect(QueryReceiver{{&value, &error}, &environment});
        assert(iterations == 0u && observations == 0);
        operation.start();
        assert(value && !error && iterations == 3u && observations == 6);

        for(unsigned failureAt : {1u, 3u})
        {
            unsigned attempts = 0u;
            auto failing = caravan::repeatUntil(
                [&]
                {
                    if(++attempts == failureAt)
                        throw std::runtime_error("repeat factory");
                    return caravan::InlineScheduler{}.schedule() | caravan::then([] { return false; });
                });
            try
            {
                caravan::syncWait(std::move(failing));
                assert(false);
            }
            catch(std::runtime_error const& failure)
            {
                assert(std::string_view{failure.what()} == "repeat factory" && attempts == failureAt);
            }
        }

        try
        {
            caravan::syncWait(
                caravan::repeatUntil([] { return ThrowingConnectSender{} | caravan::then([] { return true; }); }));
            assert(false);
        }
        catch(std::runtime_error const& failure)
        {
            assert(std::string_view{failure.what()} == "connect failed");
        }

        for(bool asynchronous : {false, true})
        {
            caravan::EventSource failed;
            auto expected = std::make_exception_ptr(std::runtime_error("repeat child"));
            if(!asynchronous)
                failed.setFailed(expected);
            unsigned attempts = 0u;
            caravan::AsyncScope failureScope;
            auto failure = failureScope.spawn(
                caravan::repeatUntil(
                    [&]
                    {
                        return caravan::asSender(++attempts == 1u ? caravan::readyEvent() : failed.event())
                               | caravan::then([] { return false; });
                    }));
            if(asynchronous)
            {
                assert(failure.state() == caravan::CompletionState::pending && attempts == 2u);
                failed.setFailed(expected);
            }
            failureScope.join().wait();
            assert(failure.state() == caravan::CompletionState::failed && failure.error() == expected);
            assert(attempts == 2u);
        }

        // Alternate completion threads so completion can race with start() returning on the other thread.
        std::array<caravan::RunLoop, 2u> loops;
        std::thread first([&] { loops[0].run(); });
        std::thread second([&] { loops[1].run(); });
        unsigned steps = 0u;
        caravan::AsyncScope threadedScope;
        auto threaded = threadedScope.spawn(
            caravan::repeatUntil(
                [&]
                {
                    auto const step = ++steps;
                    return loops[step % 2u].scheduler().schedule() | caravan::then([step] { return step == 10000u; });
                }));
        threadedScope.join().wait();
        loops[0].finish();
        loops[1].finish();
        first.join();
        second.join();
        assert(threaded.isReady() && steps == 10000u);
    }

    void testTypedSenderVocabulary()
    {
        static_assert(!caravan::Sender<UnsupportedMultiValueSender>);
        static_assert(!caravan::Sender<UnsupportedErrorSender>);
        static_assert(!caravan::Sender<UnsupportedReferenceSender>);
        static_assert(caravan::Sender<caravan::EventSender>);
        static_assert(caravan::SenderTo<caravan::EventSender, EventReceiver>);
        static_assert(caravan::Sender<AsyncValueSender<int>>);

        int environmentObservations = 0;
        int environmentResult = 0;
        auto environmentChain
            = caravan::whenAll(
                  EnvironmentSender{20, &environmentObservations},
                  EnvironmentSender{21, &environmentObservations})
              | caravan::letValue([&environmentObservations](int left, int right)
                                  { return EnvironmentSender{left + right, &environmentObservations}; })
              | caravan::continuesOn(caravan::InlineScheduler{}) | caravan::then([](int value) { return value + 1; });
        auto environmentOperation = std::move(environmentChain).connect(EnvironmentReceiver{&environmentResult});
        assert(environmentObservations == 20);
        environmentOperation.start();
        assert(environmentObservations == 30 && environmentResult == 42);

        caravan::EventSource thenReady;
        auto doubled
            = AsyncValueSender<int>{thenReady.event(), 21} | caravan::then([](int value) { return value * 2; });
        static_assert(
            std::is_same_v<
                caravan::CompletionSignaturesOf<decltype(doubled)>,
                caravan::
                    CompletionSignatures<caravan::ValueSignature<int>, caravan::ErrorSignature<std::exception_ptr>>>);
        caravan::AsyncScope thenScope;
        auto doubledResult = thenScope.spawnFuture<int>(std::move(doubled));
        thenReady.setReady();
        assert(doubledResult.result() == 42);
        thenScope.join().wait();

        bool voidThenCalled = false;
        caravan::syncWait(caravan::asSender(caravan::readyEvent()) | caravan::then([&] { voidThenCalled = true; }));
        assert(voidThenCalled);

        caravan::EventSource predecessorReady;
        caravan::EventSource successorReady;
        bool factoryCalled = false;
        auto chained = AsyncValueSender<int>{predecessorReady.event(), 20}
                       | caravan::letValue(
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
        auto combined
            = caravan::whenAll(
                  AsyncValueSender<int>{firstReady.event(), 40},
                  AsyncValueSender<std::string>{secondReady.event(), "ok"})
              | caravan::then([](int value, std::string text) { return value + static_cast<int>(text.size()); });
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
        auto failing = caravan::whenAll(
                           AsyncValueSender<int>{failedReady.event(), 1},
                           AsyncValueSender<int>{unfinishedReady.event(), 2})
                       | caravan::then([&](int, int) { continuationCalled = true; });
        caravan::AsyncScope failureScope;
        auto failed = failureScope.spawn(std::move(failing));
        failedReady.setFailed(std::make_exception_ptr(std::runtime_error("typed whenAll failure")));
        assert(failed.state() == caravan::CompletionState::pending);
        unfinishedReady.setReady();
        assert(failed.state() == caravan::CompletionState::failed);
        assert(!continuationCalled);
        failureScope.join().wait();
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

        caravan::EventSource predecessor;
        caravan::EventSource successor;
        auto retained = std::make_shared<int>(7);
        std::weak_ptr<int> lifetime = retained;
        auto completion = scope.spawn(
            caravan::asSender(predecessor.event())
            | caravan::letValue([retained, ready = successor.event()] { return caravan::asSender(ready); }));
        retained.reset();
        assert(!lifetime.expired());
        predecessor.setReady();
        assert(!lifetime.expired());
        successor.setReady();
        completion.wait();
        assert(lifetime.expired());

        caravan::Promise<std::unique_ptr<int>> promise;
        auto consumedAlias = promise.future();
        auto consumed = scope.spawnFuture<std::unique_ptr<int>>(caravan::consumeAsSender(promise.future()));
        promise.setValue(std::make_unique<int>(42));
        auto consumedValue = std::move(consumed).takeResult();
        assert(*consumedValue == 42);
        try
        {
            static_cast<void>(consumedAlias.result());
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }

        caravan::Promise<int> failedPromise;
        auto failedBridge = scope.spawn(caravan::consumeAsSender(failedPromise.future()));
        failedPromise.setFailed(std::make_exception_ptr(std::runtime_error("future bridge failure")));
        assert(failedBridge.state() == caravan::CompletionState::failed);

        scope.join().wait();
    }

    void testContinuesOnRunLoop()
    {
        caravan::RunLoop runLoop;
        auto scheduler = runLoop.scheduler();
        static_assert(std::is_trivially_copyable_v<caravan::RunLoopScheduler>);
        caravan::AsyncScope scope;
        caravan::EventSource source;
        auto transferred = scope.spawn(caravan::asSender(source.event()) | caravan::continuesOn(scheduler));
        auto joined = scope.join();

        source.setReady();
        assert(transferred.state() == caravan::CompletionState::pending);
        assert(joined.state() == caravan::CompletionState::pending);
        runLoop.finish();
        runLoop.run();
        transferred.wait();
        joined.wait();
        bool rejectedValue = false;
        std::exception_ptr rejectedError;
        auto rejected = scheduler.schedule().connect(EventReceiver{&rejectedValue, &rejectedError});
        rejected.start();
        assert(!rejectedValue && rejectedError);

        caravan::RunLoop polledLoop;
        caravan::AsyncScope polledScope;
        caravan::EventSource polledSource;
        auto polled = polledScope.spawn(
            caravan::asSender(polledSource.event()) | caravan::continuesOn(polledLoop.scheduler()));
        auto polledJoin = polledScope.join();
        polledSource.setReady();
        assert(polled.state() == caravan::CompletionState::pending);
        polledLoop.runReady();
        polled.wait();
        polledJoin.wait();
    }

    void testControlContext()
    {
        caravan::ControlContext context;
        auto const controlThread = std::this_thread::get_id();
        std::thread::id completionThread;
        bool ran = false;
        caravan::EventSource backendCompletion;

        auto operation = context.spawn(
            context.onControl(caravan::asSender(backendCompletion.event()))
            | caravan::then(
                [&]
                {
                    ran = true;
                    assert(std::this_thread::get_id() == controlThread);
                    assert(std::this_thread::get_id() != completionThread);
                }));
        assert(operation.state() == caravan::CompletionState::pending);
        std::thread backend(
            [&]
            {
                completionThread = std::this_thread::get_id();
                backendCompletion.setReady();
            });
        context.wait(operation);
        backend.join();
        assert(ran);

        caravan::EventSource externalCompletion;
        std::thread external([&] { externalCompletion.setReady(); });
        context.wait(externalCompletion.event());
        external.join();

        caravan::EventSource progressed;
        std::size_t progressCalls = 0u;
        context.wait(
            progressed.event(),
            [&]
            {
                ++progressCalls;
                progressed.setReady();
            });
        assert(progressCalls == 1u);

        caravan::EventSource pending;
        bool rejected = false;
        auto checked = context.spawn(
            context.scheduler().schedule()
            | caravan::then(
                [&]
                {
                    try
                    {
                        context.wait(pending.event());
                    }
                    catch(std::logic_error const&)
                    {
                        rejected = true;
                    }
                }));
        context.wait(checked);
        assert(rejected);

        caravan::EventSource failed;
        try
        {
            context.wait(
                failed.event(),
                [&] { failed.setFailed(std::make_exception_ptr(std::runtime_error("backend failure"))); });
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }

        try
        {
            context.wait(pending.event(), [] { throw std::runtime_error("progress failure"); });
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }
        pending.setReady();
        context.runReady();
        context.wait(pending.event());
    }

    void testScheduledCompletionChannels()
    {
        using State = caravan::CompletionState;

        // This scheduler has no post(). Both stages must be lazy, and both
        // upstream channels must wait for successful scheduling before delivery.
        for(auto state : {State::ready, State::failed})
        {
            caravan::EventSource upstream;
            caravan::EventSource scheduled;
            caravan::AsyncScope scope;
            auto error = std::make_exception_ptr(std::runtime_error("upstream"));
            auto sender
                = caravan::asSender(upstream.event()) | caravan::continuesOn(EventScheduler{scheduled.event()});
            if(state == State::ready)
                upstream.setReady();
            else
                upstream.setFailed(error);
            auto result = scope.spawn(std::move(sender));
            assert(result.state() == State::pending);
            scheduled.setReady();
            assert(result.state() == state);
            if(state == State::failed)
                assert(result.error() == error);
            scope.join().wait();
        }

        // A scheduler's own error completion overrides the stored value.
        caravan::EventSource scheduled;
        caravan::AsyncScope schedulingScope;
        auto error = std::make_exception_ptr(std::runtime_error("scheduler"));
        auto retained = std::make_shared<int>(42);
        std::weak_ptr<int> lifetime = retained;
        auto result = schedulingScope.spawn(
            AsyncValueSender<std::shared_ptr<int>>{caravan::readyEvent(), std::move(retained)}
            | caravan::continuesOn(EventScheduler{scheduled.event()}));
        assert(!lifetime.expired());
        scheduled.setFailed(error);
        assert(result.state() == State::failed && result.error() == error);
        assert(lifetime.expired());
        schedulingScope.join().wait();

        caravan::RunLoop loop;
        caravan::AsyncScope scope;
        auto moved = scope.spawnFuture<std::unique_ptr<int>>(
            AsyncValueSender<std::unique_ptr<int>>{caravan::readyEvent(), std::make_unique<int>(42)}
            | caravan::continuesOn(loop.scheduler()));
        assert(moved.state() == State::pending);
        loop.runReady();
        assert(*std::move(moved).takeResult() == 42);
        scope.join().wait();

        // Success-only composition does not attempt scheduling on upstream failure.
        loop.finish();
        caravan::EventSource upstream;
        caravan::AsyncScope failureScope;
        bool called = false;
        auto original = std::make_exception_ptr(std::runtime_error("original failure"));
        auto failed = failureScope.spawn(
            caravan::asSender(upstream.event())
            | caravan::letValue([&] { return loop.scheduler().schedule() | caravan::then([&] { called = true; }); }));
        upstream.setFailed(original);
        assert(!called && failed.state() == State::failed && failed.error() == original);
        failureScope.join().wait();
    }

    void testStartsOn()
    {
        caravan::RunLoop loop;
        caravan::EventSource nativeCompletion;
        caravan::AsyncScope scope;
        bool started = false;
        std::thread::id completedOn;
        auto work = StartTrackingSender{&started}
                    | caravan::letValue(
                        [&]
                        {
                            return AsyncValueSender<std::unique_ptr<int>>{
                                nativeCompletion.event(),
                                std::make_unique<int>(42)};
                        });
        auto result = scope.spawnFuture<std::unique_ptr<int>>(
            std::move(work) | caravan::startsOn(loop.scheduler())
            | caravan::then(
                [&](std::unique_ptr<int> value)
                {
                    completedOn = std::this_thread::get_id();
                    return value;
                }));
        assert(!started);
        loop.runReady();
        assert(started && result.state() == caravan::CompletionState::pending);
        std::thread native(
            [&]
            {
                auto const thread = std::this_thread::get_id();
                nativeCompletion.setReady();
                assert(completedOn == thread); // No return to the initiating run loop.
            });
        native.join();
        assert(*std::move(result).takeResult() == 42);
        scope.join().wait();

        caravan::EventSource scheduling;
        caravan::AsyncScope failedScope;
        bool childStarted = false;
        auto failed = failedScope.spawn(
            caravan::startsOn(EventScheduler{scheduling.event()}, StartTrackingSender{&childStarted}));
        auto error = std::make_exception_ptr(std::runtime_error("start scheduling"));
        assert(!childStarted && failed.state() == caravan::CompletionState::pending);
        scheduling.setFailed(error);
        assert(!childStarted && failed.state() == caravan::CompletionState::failed);
        assert(failed.error() == error);
        failedScope.join().wait();

        // Inline completion may immediately destroy both connected operations.
        bool destroyed = false;
        caravan::AsyncScope inlineScope;
        auto inlineResult = inlineScope.spawn(
            caravan::startsOn(caravan::InlineScheduler{}, ScopeOperationTrackingSender{&destroyed}));
        assert(destroyed && inlineResult.isReady());
        inlineScope.join().wait();

        auto checkConnectFailure = [](auto sender)
        {
            caravan::AsyncScope failedScope;
            try
            {
                failedScope.spawn(std::move(sender));
                assert(false);
            }
            catch(std::runtime_error const& error)
            {
                assert(std::string_view{error.what()} == "connect failed");
            }
            failedScope.join().wait();
        };
        checkConnectFailure(caravan::startsOn(caravan::InlineScheduler{}, ThrowingConnectSender{}));
        checkConnectFailure(
            caravan::startsOn(
                caravan::InlineScheduler{},
                caravan::on(caravan::InlineScheduler{}, ThrowingConnectSender{})));
    }

    void testPlacementEnvironment()
    {
        using Unscoped = decltype(caravan::on(TaggedScheduler{1}, caravan::asSender(caravan::readyEvent())));
        static_assert(caravan::Sender<Unscoped>);
        static_assert(!caravan::SenderTo<Unscoped, EventReceiver>); // No implicit restoration scheduler.

        bool value = false;
        std::exception_ptr error;
        int observations = 0;
        QueryEnvironment environment;
        auto work = caravan::startsOn(
            TaggedScheduler{1},
            caravan::whenAll(
                QuerySender{1, &observations},
                caravan::on(TaggedScheduler{2}, QuerySender{2, &observations} | caravan::then([] {})))
                | caravan::letValue([&] { return QuerySender{1, &observations}; })
                | caravan::continuesOn(caravan::InlineScheduler{}));
        auto operation = std::move(work).connect(QueryReceiver{{&value, &error}, &environment});
        assert(observations == 2 && !value);
        operation.start();
        assert(observations == 6 && value && !error);
    }

    void testOnRestoration()
    {
        // Model A -> M -> B with independent native completions. Inline restoration
        // submits B directly on M completion; application restoration deliberately hops.
        caravan::RunLoop app;
        auto check = [&](auto ambient, bool inlineRestore)
        {
            caravan::RunLoop mpi;
            caravan::EventSource a;
            caravan::EventSource m;
            caravan::EventSource b;
            caravan::AsyncScope scope;
            bool mpiStarted = false;
            bool bStarted = false;
            std::thread::id mpiThread;
            std::thread::id bSubmittedOn;
            std::thread::id bCompletedOn;
            auto result = scope.spawn(
                caravan::startsOn(
                    ambient,
                    caravan::asSender(a.event())
                        | caravan::letValue(
                            [&]
                            {
                                return caravan::on(
                                    mpi.scheduler(),
                                    StartTrackingSender{&mpiStarted}
                                        | caravan::letValue([&] { return caravan::asSender(m.event()); }));
                            })
                        | caravan::letValue(
                            [&]
                            {
                                bStarted = true;
                                bSubmittedOn = std::this_thread::get_id();
                                return caravan::asSender(b.event());
                            })
                        | caravan::then([&] { bCompletedOn = std::this_thread::get_id(); })));
            app.runReady();
            std::thread aThread([&] { a.setReady(); });
            aThread.join();
            assert(!mpiStarted && !bStarted);
            std::thread mpiDriver(
                [&]
                {
                    mpiThread = std::this_thread::get_id();
                    mpi.runReady();
                    assert(mpiStarted);
                    m.setReady();
                });
            mpiDriver.join();
            assert(bStarted == inlineRestore);
            app.runReady();
            assert(bStarted);
            assert(bSubmittedOn == (inlineRestore ? mpiThread : std::this_thread::get_id()));
            assert(result.state() == caravan::CompletionState::pending);
            std::thread bThread(
                [&]
                {
                    b.setReady();
                    assert(bCompletedOn == std::this_thread::get_id());
                });
            bThread.join();
            result.wait();
            scope.join().wait();
        };
        check(caravan::InlineScheduler{}, true);
        check(app.scheduler(), false);

        caravan::RunLoop middle;
        caravan::RunLoop inner;
        caravan::AsyncScope scope;
        std::vector<int> order;
        auto result = scope.spawn(
            caravan::startsOn(
                app.scheduler(),
                caravan::on(
                    middle.scheduler(),
                    (caravan::asSender(caravan::readyEvent()) | caravan::then([&] { order.push_back(1); })
                     | caravan::on(inner.scheduler()))
                        | caravan::then([&] { order.push_back(2); }))
                    | caravan::then([&] { order.push_back(3); })));
        app.runReady();
        middle.runReady();
        assert(order.empty());
        inner.runReady();
        assert((order == std::vector{1}));
        middle.runReady();
        assert((order == std::vector{1, 2}));
        app.runReady();
        assert((order == std::vector{1, 2, 3}) && result.isReady());
        scope.join().wait();
    }

    void testOnCompletionChannels()
    {
        using State = caravan::CompletionState;
        for(auto state : {State::ready, State::failed})
        {
            caravan::RunLoop app;
            caravan::EventSource native;
            caravan::AsyncScope scope;
            auto error = std::make_exception_ptr(std::runtime_error("native"));
            auto result = scope.spawnFuture<std::unique_ptr<int>>(caravan::startsOn(
                app.scheduler(),
                caravan::on(
                    caravan::InlineScheduler{},
                    AsyncValueSender<std::unique_ptr<int>>{native.event(), std::make_unique<int>(42)})));
            app.runReady();
            if(state == State::ready)
                native.setReady();
            else
                native.setFailed(error);
            assert(result.state() == State::pending);
            app.runReady();
            assert(result.state() == state);
            if(state == State::ready)
                assert(*std::move(result).takeResult() == 42);
            else
                assert(result.event().error() == error);
            scope.join().wait();
        }

        caravan::RunLoop schedulingApp;
        caravan::EventSource scheduling;
        caravan::AsyncScope schedulingScope;
        bool started = false;
        auto schedulingError = std::make_exception_ptr(std::runtime_error("initial scheduling"));
        auto schedulingResult = schedulingScope.spawn(
            caravan::startsOn(
                schedulingApp.scheduler(),
                caravan::on(EventScheduler{scheduling.event()}, StartTrackingSender{&started})));
        schedulingApp.runReady();
        scheduling.setFailed(schedulingError);
        assert(!started && schedulingResult.state() == State::pending);
        schedulingApp.runReady();
        assert(schedulingResult.state() == State::failed && schedulingResult.error() == schedulingError);
        schedulingScope.join().wait();

        // Explicit receiver environments work without an outer startsOn.
        caravan::EventSource restore;
        bool value = false;
        std::exception_ptr restorationError;
        auto operation
            = caravan::on(caravan::InlineScheduler{}, caravan::asSender(caravan::readyEvent()))
                  .connect(
                      SchedulerReceiver<EventScheduler>{{&value, &restorationError}, EventScheduler{restore.event()}});
        operation.start();
        assert(!value && !restorationError);
        auto failure = std::make_exception_ptr(std::runtime_error("restoration"));
        restore.setFailed(failure);
        assert(!value && restorationError == failure);

        // Failure while storing a value for restoration must also take the return
        // hop, rather than bypassing the requested scheduler.
        {
            caravan::RunLoop app;
            caravan::AsyncScope scope;
            auto result = scope.spawn(
                caravan::startsOn(
                    app.scheduler(),
                    caravan::on(
                        caravan::InlineScheduler{},
                        caravan::InlineScheduler{}.schedule() | caravan::then([] { return ThrowOnMove{}; }))
                        | caravan::then([](ThrowOnMove) { assert(false); })));
            app.runReady();
            assert(result.state() == State::pending);
            app.runReady();
            assert(result.state() == State::failed);
            try
            {
                result.wait();
                assert(false);
            }
            catch(std::runtime_error const& error)
            {
                assert(std::string_view{error.what()} == "value storage");
            }
            scope.join().wait();
        }

        // A finished restoration loop replaces the native error, rather than
        // delivering the original error with a false promise of thread affinity.
        caravan::RunLoop app;
        caravan::EventSource native;
        caravan::AsyncScope scope;
        auto original = std::make_exception_ptr(std::runtime_error("original"));
        auto result = scope.spawn(
            caravan::startsOn(
                app.scheduler(),
                caravan::on(caravan::InlineScheduler{}, caravan::asSender(native.event()))));
        app.runReady();
        app.finish();
        native.setFailed(original);
        assert(result.state() == State::failed && result.error() != original);
        scope.join().wait();
    }

    void testAsyncScope()
    {
        caravan::AsyncScope scope;
        assert(scope.status() == caravan::AsyncScopeStatus::open);
        caravan::EventSource readySource;
        caravan::EventSource failedSource;
        auto ready = scope.spawn(caravan::asSender(readySource.event()));
        auto failed = scope.spawn(caravan::asSender(failedSource.event()));
        auto joined = scope.join();

        assert(scope.status() == caravan::AsyncScopeStatus::joining);
        assert(joined.state() == caravan::CompletionState::pending);
        readySource.setReady();
        failedSource.setFailed(std::make_exception_ptr(std::runtime_error("scope failure")));
        joined.wait();
        assert(scope.status() == caravan::AsyncScopeStatus::joined);
        assert(ready.isReady());
        assert(failed.state() == caravan::CompletionState::failed && failed.error());

        try
        {
            scope.spawn(caravan::asSender(caravan::readyEvent()));
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }

        bool operationDestroyed = false;
        caravan::AsyncScope synchronousScope;
        auto synchronous = synchronousScope.spawn(ScopeOperationTrackingSender{&operationDestroyed});
        assert(operationDestroyed);
        synchronous.wait();
        synchronousScope.join().wait();

        caravan::AsyncScope constructionFailureScope;
        try
        {
            constructionFailureScope.spawn(ThrowingConnectSender{});
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }
        constructionFailureScope.join().wait();
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
                    scope.spawn(caravan::asSender(caravan::readyEvent()) | caravan::continuesOn(loop.scheduler())));
            }
            std::_Exit(0);
        }

        int status = 0;
        auto const waitedChild = waitpid(child, &status, 0);
        assert(waitedChild == child);
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
        caravan::InlineScheduler executor;
        caravan::EventSource source;
        caravan::AsyncScope scope;
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
                        local.emplace_back(scope.spawn(
                            caravan::asSender(source.event()) | caravan::continuesOn(executor)
                            | caravan::then([&] { calls.fetch_add(1u); })));
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
        scope.join().wait();
    }

    void testExecutorWaitGuard()
    {
        caravan::InlineScheduler executor;
        caravan::EventSource start;
        caravan::EventSource pending;
        caravan::AsyncScope scope;
        auto blocked = scope.spawn(
            caravan::asSender(start.event()) | caravan::continuesOn(executor)
            | caravan::then([&] { pending.event().wait(); }));
        start.setReady();
        try
        {
            blocked.wait();
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }
        scope.join().wait();
    }

    void testDispatchRecoversFromThrow()
    {
        try
        {
            caravan::detail::dispatch(
                std::make_unique<caravan::detail::DispatchTask>([] { throw std::runtime_error("dispatch failure"); }));
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }

        bool ran = false;
        caravan::detail::dispatch(std::make_unique<caravan::detail::DispatchTask>([&] { ran = true; }));
        assert(ran);
    }
} // namespace

int main()
{
    assert(caravan::readyEvent().isReady());
    testInlineScheduler();
    testCompletionAndContinuations();
    testRunReadyFairness();
    testSchedulerHandleLifetime();
    testNoRecursiveInlineChains();
    testWhenAllAndFailure();
    testFuture();
    testEventSenderBridge();
    testSyncWait();
    testLetValue();
    testSequence();
    testRepeatUntil();
    testTypedSenderVocabulary();
    testEagerSenderBridgesAndOperationLifetime();
    testContinuesOnRunLoop();
    testControlContext();
    testScheduledCompletionChannels();
    testStartsOn();
    testPlacementEnvironment();
    testOnRestoration();
    testOnCompletionChannels();
    testAsyncScope();
    testPendingScopeDestructionDiagnosed();
    testExactlyOnceCompletion();
    testRegistrationRace();
    testExecutorWaitGuard();
    testDispatchRecoversFromThrow();
}
