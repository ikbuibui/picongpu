/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <caravan/core.hpp>

namespace caravan
{
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

    /** Manually driven single-thread executor for host/control work. */
    class RunLoop
    {
    public:
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
        std::mutex m_mutex;
        std::condition_variable m_ready;
        std::deque<std::function<void()>> m_tasks;
        bool m_finished = false;
    };

    namespace detail
    {
        template<typename T_Sender, typename T_Executor, typename T_Receiver>
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
            ContinuesOnOperation(T_Sender sender, T_Executor& executor, T_Receiver receiver)
                : m_executor(&executor)
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
                    m_executor->post(
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
                    m_executor->post([this, error = std::move(error)]() mutable
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
                    m_executor->post([this] { m_receiver.set_stopped(); });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            T_Executor* m_executor;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<TransferReceiver>())) m_upstream;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Executor>
    class ContinuesOnSender
    {
    public:
        ContinuesOnSender(T_Sender sender, T_Executor& executor) : m_sender(std::move(sender)), m_executor(&executor)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ContinuesOnOperation<T_Sender, T_Executor, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                *m_executor,
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Executor* m_executor;
    };

    /** Transfer sender completion onto an explicit executor. */
    template<typename T_Sender, typename T_Executor>
    auto continuesOn(T_Sender sender, T_Executor& executor)
    {
        return ContinuesOnSender<T_Sender, T_Executor>{std::move(sender), executor};
    }

    namespace detail
    {
        template<typename T_Sender, typename T_Factory, typename T_Receiver>
        class LetValueOperation
        {
            struct PredecessorReceiver
            {
                void set_value() noexcept
                {
                    owner->startSuccessor();
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

            class SuccessorOperation
            {
            public:
                virtual ~SuccessorOperation() = default;
                virtual void start() noexcept = 0;
            };

            using SuccessorSender = std::invoke_result_t<T_Factory&>;

            class SuccessorOperationModel final : public SuccessorOperation
            {
            public:
                SuccessorOperationModel(SuccessorSender sender, T_Receiver receiver)
                    : m_operation(std::move(sender).connect(std::move(receiver)))
                {
                }

                void start() noexcept override
                {
                    m_operation.start();
                }

            private:
                decltype(std::declval<SuccessorSender&&>().connect(std::declval<T_Receiver>())) m_operation;
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
            void startSuccessor() noexcept
            {
                try
                {
                    auto successor = std::invoke(m_factory);
                    m_successor
                        = std::make_unique<SuccessorOperationModel>(std::move(successor), std::move(m_receiver));
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
            std::unique_ptr<SuccessorOperation> m_successor;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Factory>
    class LetValueSender
    {
    public:
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

    /** Lazily create a successor after a void predecessor succeeds. */
    template<typename T_Sender, typename T_Factory>
    auto letValue(T_Sender sender, T_Factory factory)
    {
        // ponytail: void-only until a value-carrying chain is needed; replace with standard let_value.
        return LetValueSender<T_Sender, T_Factory>{std::move(sender), std::move(factory)};
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
