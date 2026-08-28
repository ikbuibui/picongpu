/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <concepts>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <caravan/core.hpp>

namespace caravan
{
    template<typename... T>
    struct ValueSignature
    {
    };

    template<typename T>
    struct ErrorSignature
    {
    };

    struct StoppedSignature
    {
    };

    template<typename... T_Signatures>
    struct CompletionSignatures
    {
    };

    template<typename T_Sender>
    using CompletionSignaturesOf = typename std::remove_cvref_t<T_Sender>::completion_signatures;

    template<typename T>
    inline constexpr bool isCompletionSignatures = false;

    template<typename... T_Signatures>
    inline constexpr bool isCompletionSignatures<CompletionSignatures<T_Signatures...>> = true;

    template<typename T_Sender>
    concept Sender = requires { typename CompletionSignaturesOf<T_Sender>; }
                     && isCompletionSignatures<CompletionSignaturesOf<T_Sender>>;

    template<typename T_Operation>
    concept OperationState = requires(T_Operation& operation) {
        { operation.start() } noexcept;
    };

    template<typename T_Sender, typename T_Receiver>
    concept SenderTo = Sender<T_Sender> && requires(T_Sender&& sender, T_Receiver&& receiver) {
        { std::forward<T_Sender>(sender).connect(std::forward<T_Receiver>(receiver)) } -> OperationState;
    };

    namespace detail
    {
        template<typename T_Signatures>
        struct ValueTuple;

        template<typename... T, typename... T_Rest>
        struct ValueTuple<CompletionSignatures<ValueSignature<T...>, T_Rest...>>
        {
            using type = std::tuple<T...>;
        };

        template<typename T_First, typename... T_Rest>
        struct ValueTuple<CompletionSignatures<T_First, T_Rest...>> : ValueTuple<CompletionSignatures<T_Rest...>>
        {
        };

        template<typename T_Sender>
        using ValueTupleOf = typename ValueTuple<CompletionSignaturesOf<T_Sender>>::type;

        template<typename T_Tuple>
        struct ValueSignatureFromTuple;

        template<typename... T>
        struct ValueSignatureFromTuple<std::tuple<T...>>
        {
            using type = ValueSignature<T...>;
        };

        template<typename T_Function, typename T_Tuple>
        struct InvokeResultFromTuple;

        template<typename T_Function, typename... T>
        struct InvokeResultFromTuple<T_Function, std::tuple<T...>>
        {
            using type = std::invoke_result_t<T_Function&, T...>;
        };

        template<typename T_Result>
        using ResultValueSignature = std::
            conditional_t<std::is_void_v<T_Result>, ValueSignature<>, ValueSignature<std::remove_cvref_t<T_Result>>>;

        template<typename T_ValueSignature>
        using DefaultCompletionSignatures
            = CompletionSignatures<T_ValueSignature, ErrorSignature<std::exception_ptr>, StoppedSignature>;

        template<typename T_Sender, typename T_Function>
        using ThenCompletionSignatures = DefaultCompletionSignatures<
            ResultValueSignature<typename InvokeResultFromTuple<T_Function, ValueTupleOf<T_Sender>>::type>>;

        template<typename T_Sender, typename T_Function>
        using SuccessorSender = typename InvokeResultFromTuple<T_Function, ValueTupleOf<T_Sender>>::type;

        template<typename... T_Senders>
        using CombinedValueTuple = decltype(std::tuple_cat(std::declval<ValueTupleOf<T_Senders>>()...));
    } // namespace detail

    /** A lazy sender bridge for an already-started Event.
     *
     * The connected operation must remain at a stable address from start until
     * receiver completion, matching the P2300 operation-state lifetime rule.
     */
    template<typename T_Receiver>
    class EventOperation
    {
    public:
        EventOperation(Event event, T_Receiver receiver) : m_event(std::move(event)), m_receiver(std::move(receiver))
        {
        }

        EventOperation(EventOperation const&) = delete;
        EventOperation& operator=(EventOperation const&) = delete;
        EventOperation(EventOperation&&) = delete;
        EventOperation& operator=(EventOperation&&) = delete;

        void start() & noexcept
        {
            m_event.subscribe([this] { complete(); });
        }

    private:
        void complete() noexcept
        {
            switch(m_event.state())
            {
            case CompletionState::ready:
                m_receiver.set_value();
                break;
            case CompletionState::failed:
                m_receiver.set_error(m_event.error());
                break;
            case CompletionState::stopped:
                m_receiver.set_stopped();
                break;
            case CompletionState::pending:
                std::terminate();
            }
        }

        Event m_event;
        T_Receiver m_receiver;
    };

    class EventSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit EventSender(Event event) : m_event(std::move(event))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) const
        {
            return EventOperation<std::decay_t<T_Receiver>>{m_event, std::forward<T_Receiver>(receiver)};
        }

    private:
        Event m_event;
    };

    inline EventSender asSender(Event event)
    {
        return EventSender{std::move(event)};
    }

    class RunLoopScheduler;

    /** Manually driven single-thread queue for host/control work. */
    class RunLoop
    {
    public:
        RunLoopScheduler scheduler() noexcept;

        void run()
        {
            ExecutorThreadGuard guard;
            for(;;)
            {
                std::function<void()> task;
                {
                    std::unique_lock lock(m_mutex);
                    m_ready.wait(lock, [this] { return m_finished || !m_tasks.empty(); });
                    if(m_tasks.empty())
                        return;
                    task = std::move(m_tasks.front());
                    m_tasks.pop_front();
                }
                task();
            }
        }

        /** Execute all currently ready work without blocking. */
        void runReady()
        {
            ExecutorThreadGuard guard;
            for(;;)
            {
                std::function<void()> task;
                {
                    std::lock_guard lock(m_mutex);
                    if(m_tasks.empty())
                        return;
                    task = std::move(m_tasks.front());
                    m_tasks.pop_front();
                }
                task();
            }
        }

        void finish()
        {
            {
                std::lock_guard lock(m_mutex);
                m_finished = true;
            }
            m_ready.notify_all();
        }

    private:
        template<typename T_Function>
        void post(T_Function&& function)
        {
            {
                std::lock_guard lock(m_mutex);
                if(m_finished)
                    throw std::logic_error("Cannot post to a finished Caravan run loop");
                m_tasks.emplace_back(std::forward<T_Function>(function));
            }
            m_ready.notify_one();
        }

        std::mutex m_mutex;
        std::condition_variable m_ready;
        std::deque<std::function<void()>> m_tasks;
        bool m_finished = false;

        friend class RunLoopScheduler;
    };

    /** Cheap scheduling handle; its RunLoop must outlive it. */
    class RunLoopScheduler
    {
    public:
        template<typename T_Function>
        void post(T_Function&& function) const
        {
            m_loop->post(std::forward<T_Function>(function));
        }

    private:
        explicit RunLoopScheduler(RunLoop& loop) : m_loop(&loop)
        {
        }

        RunLoop* m_loop;

        friend class RunLoop;
    };

    inline RunLoopScheduler RunLoop::scheduler() noexcept
    {
        return RunLoopScheduler{*this};
    }

    namespace detail
    {
        template<typename T_Sender, typename T_Scheduler, typename T_Receiver>
        class ContinuesOnOperation
        {
            struct TransferReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->transferValue(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->transferError(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->transferStopped();
                }

                ContinuesOnOperation* owner;
            };

        public:
            ContinuesOnOperation(T_Sender sender, T_Scheduler scheduler, T_Receiver receiver)
                : m_scheduler(std::move(scheduler))
                , m_receiver(std::move(receiver))
                , m_upstream(std::move(sender).connect(TransferReceiver{this}))
            {
            }

            ContinuesOnOperation(ContinuesOnOperation const&) = delete;
            ContinuesOnOperation& operator=(ContinuesOnOperation const&) = delete;
            ContinuesOnOperation(ContinuesOnOperation&&) = delete;
            ContinuesOnOperation& operator=(ContinuesOnOperation&&) = delete;

            void start() & noexcept
            {
                m_upstream.start();
            }

        private:
            template<typename... T>
            void transferValue(T&&... values) noexcept
            {
                try
                {
                    auto storedValues = std::make_shared<std::tuple<std::decay_t<T>...>>(std::forward<T>(values)...);
                    m_scheduler.post(
                        [this, storedValues = std::move(storedValues)]() mutable
                        {
                            std::apply(
                                [this](auto&&... unpacked)
                                { m_receiver.set_value(std::forward<decltype(unpacked)>(unpacked)...); },
                                std::move(*storedValues));
                        });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            void transferError(std::exception_ptr error) noexcept
            {
                try
                {
                    m_scheduler.post([this, error = std::move(error)]() mutable
                                     { m_receiver.set_error(std::move(error)); });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            void transferStopped() noexcept
            {
                try
                {
                    m_scheduler.post([this] { m_receiver.set_stopped(); });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            T_Scheduler m_scheduler;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<TransferReceiver>())) m_upstream;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Scheduler>
    class ContinuesOnSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<T_Sender>;

        ContinuesOnSender(T_Sender sender, T_Scheduler scheduler)
            : m_sender(std::move(sender))
            , m_scheduler(std::move(scheduler))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ContinuesOnOperation<T_Sender, T_Scheduler, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_scheduler),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Scheduler m_scheduler;
    };

    /** Transfer sender completion onto an explicit scheduler. */
    template<Sender T_Sender, typename T_Scheduler>
    auto continuesOn(T_Sender sender, T_Scheduler scheduler)
    {
        return ContinuesOnSender<T_Sender, T_Scheduler>{std::move(sender), std::move(scheduler)};
    }

    namespace detail
    {
        template<typename T_Sender, typename T_Function, typename T_Receiver>
        class ThenOperation
        {
            struct PredecessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->completeValue(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->m_receiver.set_error(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->m_receiver.set_stopped();
                }

                ThenOperation* owner;
            };

        public:
            ThenOperation(T_Sender sender, T_Function function, T_Receiver receiver)
                : m_function(std::move(function))
                , m_receiver(std::move(receiver))
                , m_predecessor(std::move(sender).connect(PredecessorReceiver{this}))
            {
            }

            ThenOperation(ThenOperation const&) = delete;
            ThenOperation& operator=(ThenOperation const&) = delete;
            ThenOperation(ThenOperation&&) = delete;
            ThenOperation& operator=(ThenOperation&&) = delete;

            void start() & noexcept
            {
                m_predecessor.start();
            }

        private:
            template<typename... T>
            void completeValue(T&&... values) noexcept
            {
                try
                {
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Function&, T...>>)
                    {
                        std::invoke(m_function, std::forward<T>(values)...);
                        m_receiver.set_value();
                    }
                    else
                        m_receiver.set_value(std::invoke(m_function, std::forward<T>(values)...));
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            T_Function m_function;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<PredecessorReceiver>())) m_predecessor;
        };

        template<typename T_Sender, typename T_Factory, typename T_Receiver>
        class LetValueOperation
        {
            struct PredecessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->startSuccessor(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->m_receiver.set_error(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->m_receiver.set_stopped();
                }

                LetValueOperation* owner;
            };

            struct SuccessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->m_receiver.set_value(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->m_receiver.set_error(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->m_receiver.set_stopped();
                }

                LetValueOperation* owner;
            };

            using SuccessorSenderType = SuccessorSender<T_Sender, T_Factory>;

            class SuccessorOperation
            {
            public:
                SuccessorOperation(SuccessorSenderType sender, LetValueOperation* owner)
                    : m_operation(std::move(sender).connect(SuccessorReceiver{owner}))
                {
                }

                void start() noexcept
                {
                    m_operation.start();
                }

            private:
                decltype(std::declval<SuccessorSenderType&&>().connect(std::declval<SuccessorReceiver>())) m_operation;
            };

        public:
            LetValueOperation(T_Sender sender, T_Factory factory, T_Receiver receiver)
                : m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
                , m_predecessor(std::move(sender).connect(PredecessorReceiver{this}))
            {
            }

            LetValueOperation(LetValueOperation const&) = delete;
            LetValueOperation& operator=(LetValueOperation const&) = delete;
            LetValueOperation(LetValueOperation&&) = delete;
            LetValueOperation& operator=(LetValueOperation&&) = delete;

            void start() & noexcept
            {
                m_predecessor.start();
            }

        private:
            template<typename... T>
            void startSuccessor(T&&... values) noexcept
            {
                try
                {
                    m_successor.emplace(std::invoke(m_factory, std::forward<T>(values)...), this);
                    m_successor->start();
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            T_Factory m_factory;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<PredecessorReceiver>())) m_predecessor;
            std::optional<SuccessorOperation> m_successor;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Function>
    class ThenSender
    {
    public:
        using completion_signatures = detail::ThenCompletionSignatures<T_Sender, T_Function>;

        ThenSender(T_Sender sender, T_Function function) : m_sender(std::move(sender)), m_function(std::move(function))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ThenOperation<T_Sender, T_Function, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_function),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Function m_function;
    };

    template<Sender T_Sender, typename T_Function>
    auto then(T_Sender sender, T_Function function)
    {
        return ThenSender<T_Sender, T_Function>{std::move(sender), std::move(function)};
    }

    template<typename T_Sender, typename T_Factory>
    class LetValueSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<detail::SuccessorSender<T_Sender, T_Factory>>;

        LetValueSender(T_Sender sender, T_Factory factory) : m_sender(std::move(sender)), m_factory(std::move(factory))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::LetValueOperation<T_Sender, T_Factory, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_factory),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Factory m_factory;
    };

    template<Sender T_Sender, typename T_Factory>
    auto letValue(T_Sender sender, T_Factory factory)
    {
        return LetValueSender<T_Sender, T_Factory>{std::move(sender), std::move(factory)};
    }

    namespace detail
    {
        template<std::size_t T_Index, typename T_Owner>
        struct WhenAllReceiver
        {
            template<typename... T>
            void set_value(T&&... values) noexcept
            {
                owner->template setValue<T_Index>(std::forward<T>(values)...);
            }

            void set_error(std::exception_ptr error) noexcept
            {
                owner->setError(std::move(error));
            }

            void set_stopped() noexcept
            {
                owner->setStopped();
            }

            T_Owner* owner;
        };

        template<std::size_t T_Index, typename T_Owner, typename T_Sender>
        class WhenAllOperationHolder
        {
            using Receiver = WhenAllReceiver<T_Index, T_Owner>;

        public:
            WhenAllOperationHolder(T_Sender sender, T_Owner* owner)
                : m_operation(std::move(sender).connect(Receiver{owner}))
            {
            }

            void start() noexcept
            {
                m_operation.start();
            }

        private:
            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };

        template<typename T_Receiver, typename T_Indices, typename... T_Senders>
        class WhenAllOperation;

        template<typename T_Receiver, std::size_t... T_Index, typename... T_Senders>
        class WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>
            : private WhenAllOperationHolder<
                  T_Index,
                  WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>,
                  T_Senders>...
        {
            using Self = WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>;

            template<std::size_t T_I>
            using Holder = WhenAllOperationHolder<T_I, Self, std::tuple_element_t<T_I, std::tuple<T_Senders...>>>;

        public:
            WhenAllOperation(std::tuple<T_Senders...> senders, T_Receiver receiver)
                : Holder<T_Index>(std::move(std::get<T_Index>(senders)), this)...
                , m_receiver(std::move(receiver))
            {
            }

            WhenAllOperation(WhenAllOperation const&) = delete;
            WhenAllOperation& operator=(WhenAllOperation const&) = delete;
            WhenAllOperation(WhenAllOperation&&) = delete;
            WhenAllOperation& operator=(WhenAllOperation&&) = delete;

            void start() & noexcept
            {
                if constexpr(sizeof...(T_Senders) == 0u)
                    m_receiver.set_value();
                else
                    (Holder<T_Index>::start(), ...);
            }

            template<std::size_t T_I, typename... T>
            void setValue(T&&... values) noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    try
                    {
                        std::get<T_I>(m_values).emplace(std::forward<T>(values)...);
                    }
                    catch(...)
                    {
                        if(!m_error)
                            m_error = std::current_exception();
                    }
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

            void setError(std::exception_ptr error) noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    if(!m_error)
                        m_error = std::move(error);
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

            void setStopped() noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    m_stopped = true;
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

        private:
            void finish() noexcept
            {
                if(m_error)
                {
                    m_receiver.set_error(std::move(m_error));
                    return;
                }
                if(m_stopped)
                {
                    m_receiver.set_stopped();
                    return;
                }

                try
                {
                    auto values
                        = std::apply([](auto&... value) { return std::tuple_cat(std::move(*value)...); }, m_values);
                    std::apply(
                        [this](auto&&... value) { m_receiver.set_value(std::forward<decltype(value)>(value)...); },
                        std::move(values));
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            T_Receiver m_receiver;
            std::mutex m_mutex;
            std::size_t m_remaining = sizeof...(T_Senders);
            std::tuple<std::optional<ValueTupleOf<T_Senders>>...> m_values;
            std::exception_ptr m_error;
            bool m_stopped = false;
        };
    } // namespace detail

    template<typename... T_Senders>
    class WhenAllSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<
            typename detail::ValueSignatureFromTuple<detail::CombinedValueTuple<T_Senders...>>::type>;

        explicit WhenAllSender(T_Senders... senders) : m_senders(std::move(senders)...)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::
                WhenAllOperation<std::decay_t<T_Receiver>, std::index_sequence_for<T_Senders...>, T_Senders...>{
                    std::move(m_senders),
                    std::forward<T_Receiver>(receiver)};
        }

    private:
        std::tuple<T_Senders...> m_senders;
    };

    template<typename... T_Senders>
    requires(Sender<T_Senders> && ...)
    auto whenAll(T_Senders... senders)
    {
        return WhenAllSender<T_Senders...>{std::move(senders)...};
    }

    namespace detail
    {
        template<typename T>
        struct SyncWaitReceiver
        {
            template<typename U>
            void set_value(U&& value) noexcept
            {
                try
                {
                    output.setValue(std::forward<U>(value));
                }
                catch(...)
                {
                    output.setFailed(std::current_exception());
                }
            }

            void set_error(std::exception_ptr error) noexcept
            {
                output.setFailed(std::move(error));
            }

            void set_stopped() noexcept
            {
                output.setStopped();
            }

            Promise<T> output;
        };

        struct SyncWaitVoidReceiver
        {
            void set_value() noexcept
            {
                output.setReady();
            }

            void set_error(std::exception_ptr error) noexcept
            {
                output.setFailed(std::move(error));
            }

            void set_stopped() noexcept
            {
                output.setStopped();
            }

            EventSource output;
        };
    } // namespace detail

    /** Start a sender and block at an imperative boundary. */
    template<typename T, typename T_Sender>
    T syncWait(T_Sender sender)
    {
        Promise<T> output;
        auto result = output.future();
        auto operation = std::move(sender).connect(detail::SyncWaitReceiver<T>{output});
        operation.start();
        return result.result();
    }

    template<typename T_Sender>
    void syncWait(T_Sender sender)
    {
        EventSource output;
        auto result = output.event();
        auto operation = std::move(sender).connect(detail::SyncWaitVoidReceiver{output});
        operation.start();
        result.wait();
    }

    namespace detail
    {
        class SpawnOperation
        {
        public:
            virtual ~SpawnOperation() = default;
            virtual void start() noexcept = 0;
        };

        class AsyncScopeState
        {
        public:
            std::size_t reserve()
            {
                std::lock_guard lock(m_mutex);
                if(m_closed)
                    throw std::logic_error("Cannot spawn into a joined Caravan async scope");
                auto const id = m_nextId++;
                m_operations.emplace(id, nullptr);
                return id;
            }

            void attach(std::size_t id, std::shared_ptr<SpawnOperation> operation)
            {
                std::lock_guard lock(m_mutex);
                m_operations.at(id) = std::move(operation);
            }

            void complete(std::size_t id) noexcept
            {
                std::shared_ptr<SpawnOperation> operation;
                bool joined = false;
                {
                    std::lock_guard lock(m_mutex);
                    auto const found = m_operations.find(id);
                    if(found == m_operations.end())
                        return;
                    operation = std::move(found->second);
                    m_operations.erase(found);
                    joined = m_closed && m_operations.empty();
                }
                if(joined)
                    m_joined.setReady();
            }

            Event join()
            {
                bool joined = false;
                {
                    std::lock_guard lock(m_mutex);
                    m_closed = true;
                    joined = m_operations.empty();
                }
                if(joined)
                    m_joined.setReady();
                return m_joined.event();
            }

        private:
            std::mutex m_mutex;
            std::unordered_map<std::size_t, std::shared_ptr<SpawnOperation>> m_operations;
            std::size_t m_nextId = 0u;
            bool m_closed = false;
            EventSource m_joined;
        };

        struct ScopeReceiver
        {
            template<typename... T>
            void set_value(T&&...) noexcept
            {
                output.setReady();
                scope->complete(id);
            }

            void set_error(std::exception_ptr error) noexcept
            {
                output.setFailed(std::move(error));
                scope->complete(id);
            }

            void set_stopped() noexcept
            {
                output.setStopped();
                scope->complete(id);
            }

            std::shared_ptr<AsyncScopeState> scope;
            EventSource output;
            std::size_t id;
        };

        template<typename T>
        struct FutureScopeReceiver
        {
            template<typename U>
            void set_value(U&& value) noexcept
            {
                try
                {
                    output.setValue(std::forward<U>(value));
                }
                catch(...)
                {
                    output.setFailed(std::current_exception());
                }
                scope->complete(id);
            }

            void set_error(std::exception_ptr error) noexcept
            {
                output.setFailed(std::move(error));
                scope->complete(id);
            }

            void set_stopped() noexcept
            {
                output.setStopped();
                scope->complete(id);
            }

            std::shared_ptr<AsyncScopeState> scope;
            Promise<T> output;
            std::size_t id;
        };

        template<typename T_Sender, typename T_Receiver>
        class SpawnOperationModel final : public SpawnOperation
        {
        public:
            SpawnOperationModel(T_Sender sender, T_Receiver receiver)
                : m_operation(std::move(sender).connect(std::move(receiver)))
            {
            }

            void start() noexcept override
            {
                m_operation.start();
            }

        private:
            decltype(std::declval<T_Sender&&>().connect(std::declval<T_Receiver>())) m_operation;
        };
    } // namespace detail

    /** Owns eagerly spawned sender operations until receiver completion.
     *
     * join() closes the scope to new work and completes when every operation is
     * quiescent. The returned Event carries each sender's value/error/stopped
     * channel; values are deliberately type-erased at this migration boundary.
     */
    class AsyncScope
    {
    public:
        AsyncScope() : m_state(std::make_shared<detail::AsyncScopeState>())
        {
        }

        AsyncScope(AsyncScope const&) = delete;
        AsyncScope& operator=(AsyncScope const&) = delete;

        ~AsyncScope()
        {
            m_state->join().wait();
        }

        template<typename T_Sender>
        Event spawn(T_Sender sender)
        {
            EventSource output;
            auto result = output.event();
            auto const id = m_state->reserve();
            try
            {
                using Receiver = detail::ScopeReceiver;
                using Operation = detail::SpawnOperationModel<T_Sender, Receiver>;
                auto operation = std::make_shared<Operation>(std::move(sender), Receiver{m_state, output, id});
                m_state->attach(id, operation);
                operation->start();
            }
            catch(...)
            {
                m_state->complete(id);
                throw;
            }
            return result;
        }

        /** Eagerly spawn a single-value sender and retain its operation state. */
        template<typename T, typename T_Sender>
        Future<T> spawnFuture(T_Sender sender)
        {
            Promise<T> output;
            auto result = output.future();
            auto const id = m_state->reserve();
            try
            {
                using Receiver = detail::FutureScopeReceiver<T>;
                using Operation = detail::SpawnOperationModel<T_Sender, Receiver>;
                auto operation = std::make_shared<Operation>(std::move(sender), Receiver{m_state, output, id});
                m_state->attach(id, operation);
                operation->start();
            }
            catch(...)
            {
                m_state->complete(id);
                throw;
            }
            return result;
        }

        Event join()
        {
            return m_state->join();
        }

    private:
        std::shared_ptr<detail::AsyncScopeState> m_state;
    };
} // namespace caravan
