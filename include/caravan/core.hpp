/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace caravan
{
    enum class CompletionState : std::uint8_t
    {
        pending,
        ready,
        failed,
        cancelled
    };

    class CancelledError : public std::runtime_error
    {
    public:
        CancelledError() : std::runtime_error("Caravan operation was cancelled")
        {
        }
    };

    class ExecutorThreadGuard
    {
    public:
        ExecutorThreadGuard();
        ~ExecutorThreadGuard();

        ExecutorThreadGuard(ExecutorThreadGuard const&) = delete;
        ExecutorThreadGuard& operator=(ExecutorThreadGuard const&) = delete;
    };

    namespace detail
    {
        inline thread_local std::size_t executorDepth = 0u;

        struct DispatchQueue
        {
            bool running = false;
            std::deque<std::function<void()>> tasks;
        };

        inline thread_local DispatchQueue dispatchQueue;

        inline void dispatch(std::function<void()> task)
        {
            dispatchQueue.tasks.emplace_back(std::move(task));
            if(dispatchQueue.running)
                return;

            dispatchQueue.running = true;
            while(!dispatchQueue.tasks.empty())
            {
                auto next = std::move(dispatchQueue.tasks.front());
                dispatchQueue.tasks.pop_front();
                next();
            }
            dispatchQueue.running = false;
        }

        class State
        {
        public:
            CompletionState get() const noexcept
            {
                return m_completion.load(std::memory_order_acquire);
            }

            std::exception_ptr error() const noexcept
            {
                return get() == CompletionState::failed ? m_error : std::exception_ptr{};
            }

            void wait() const
            {
                std::unique_lock lock(m_mutex);
                if(m_completion.load(std::memory_order_acquire) == CompletionState::pending && executorDepth != 0u)
                    throw std::logic_error("An executor thread cannot block on pending Caravan work");
                m_completed.wait(
                    lock,
                    [this] { return m_completion.load(std::memory_order_acquire) != CompletionState::pending; });
            }

            void subscribe(std::function<void()> continuation)
            {
                {
                    std::lock_guard lock(m_mutex);
                    if(m_completion.load(std::memory_order_relaxed) == CompletionState::pending)
                    {
                        m_continuations.emplace_back(std::move(continuation));
                        return;
                    }
                }
                dispatch(std::move(continuation));
            }

            bool complete(CompletionState completion, std::exception_ptr error = {})
            {
                return complete(completion, std::move(error), [] {});
            }

        protected:
            template<typename T_Prepare>
            bool complete(CompletionState completion, std::exception_ptr error, T_Prepare&& prepare)
            {
                std::vector<std::function<void()>> continuations;
                {
                    std::lock_guard lock(m_mutex);
                    if(m_completion.load(std::memory_order_relaxed) != CompletionState::pending)
                        return false;
                    std::forward<T_Prepare>(prepare)();
                    m_error = std::move(error);
                    m_completion.store(completion, std::memory_order_release);
                    continuations.swap(m_continuations);
                }
                m_completed.notify_all();
                for(auto& continuation : continuations)
                    dispatch(std::move(continuation));
                return true;
            }

        private:
            std::atomic<CompletionState> m_completion{CompletionState::pending};
            mutable std::mutex m_mutex;
            mutable std::condition_variable m_completed;
            std::exception_ptr m_error;
            std::vector<std::function<void()>> m_continuations;
        };

        template<typename T>
        class FutureState final : public State
        {
        public:
            template<typename U>
            bool setValue(U&& value)
            {
                return State::complete(
                    CompletionState::ready,
                    {},
                    [this, &value] { m_value.emplace(std::forward<U>(value)); });
            }

            T const& value() const
            {
                return *m_value;
            }

        private:
            std::optional<T> m_value;
        };
    } // namespace detail

    inline ExecutorThreadGuard::ExecutorThreadGuard()
    {
        ++detail::executorDepth;
    }

    inline ExecutorThreadGuard::~ExecutorThreadGuard()
    {
        --detail::executorDepth;
    }

    class EventSource;
    template<typename T>
    class Promise;
    template<typename T>
    class Future;

    class Event
    {
    public:
        Event() = default;

        CompletionState state() const noexcept
        {
            return m_state ? m_state->get() : CompletionState::ready;
        }

        bool isReady() const noexcept
        {
            return state() == CompletionState::ready;
        }

        std::exception_ptr error() const noexcept
        {
            return m_state ? m_state->error() : std::exception_ptr{};
        }

        void wait() const
        {
            if(m_state)
                m_state->wait();
            reportFailure();
        }

        template<typename T_Executor, typename T_Operation>
        Event then(T_Executor& executor, T_Operation&& operation) const;

    private:
        explicit Event(std::shared_ptr<detail::State> state) : m_state(std::move(state))
        {
        }

        void subscribe(std::function<void()> continuation) const
        {
            if(m_state)
                m_state->subscribe(std::move(continuation));
            else
                detail::dispatch(std::move(continuation));
        }

        void reportFailure() const
        {
            if(state() == CompletionState::failed)
                std::rethrow_exception(error());
            if(state() == CompletionState::cancelled)
                throw CancelledError{};
        }

        std::shared_ptr<detail::State> m_state;

        friend class EventSource;
        friend Event whenAll(std::span<Event const>);
        template<typename T>
        friend class Future;
    };

    class EventSource
    {
    public:
        EventSource() : m_state(std::make_shared<detail::State>())
        {
        }

        Event event() const
        {
            return Event{m_state};
        }

        bool setReady() const
        {
            return m_state->complete(CompletionState::ready);
        }

        bool setFailed(std::exception_ptr error) const
        {
            if(!error)
                error = std::make_exception_ptr(std::runtime_error("Caravan operation failed without an error"));
            return m_state->complete(CompletionState::failed, std::move(error));
        }

        bool cancel() const
        {
            return m_state->complete(CompletionState::cancelled);
        }

    private:
        std::shared_ptr<detail::State> m_state;
    };

    inline Event readyEvent()
    {
        return {};
    }

    template<typename T_Executor, typename T_Operation>
    Event Event::then(T_Executor& executor, T_Operation&& operation) const
    {
        EventSource successor;
        auto result = successor.event();
        auto predecessor = *this;
        auto work = std::make_shared<std::decay_t<T_Operation>>(std::forward<T_Operation>(operation));
        subscribe(
            [predecessor, successor, work, executor = &executor]
            {
                if(predecessor.state() == CompletionState::failed)
                {
                    successor.setFailed(predecessor.error());
                    return;
                }
                if(predecessor.state() == CompletionState::cancelled)
                {
                    successor.cancel();
                    return;
                }

                auto task = [successor, work]
                {
                    ExecutorThreadGuard guard;
                    try
                    {
                        std::invoke(*work);
                        successor.setReady();
                    }
                    catch(...)
                    {
                        successor.setFailed(std::current_exception());
                    }
                };
                try
                {
                    executor->post(std::move(task));
                }
                catch(...)
                {
                    successor.setFailed(std::current_exception());
                }
            });
        return result;
    }

    namespace detail
    {
        class WhenAllState final : public State
        {
        public:
            explicit WhenAllState(std::size_t count) : m_remaining(count)
            {
            }

            void arrive(Event const& event)
            {
                {
                    std::lock_guard lock(m_resultMutex);
                    if(m_result == CompletionState::ready && event.state() == CompletionState::failed)
                    {
                        m_result = CompletionState::failed;
                        m_error = event.error();
                    }
                    else if(m_result == CompletionState::ready && event.state() == CompletionState::cancelled)
                        m_result = CompletionState::cancelled;
                }

                if(m_remaining.fetch_sub(1u, std::memory_order_acq_rel) == 1u)
                {
                    CompletionState result;
                    std::exception_ptr error;
                    {
                        std::lock_guard lock(m_resultMutex);
                        result = m_result;
                        error = m_error;
                    }
                    complete(result, std::move(error));
                }
            }

        private:
            std::atomic<std::size_t> m_remaining;
            std::mutex m_resultMutex;
            CompletionState m_result = CompletionState::ready;
            std::exception_ptr m_error;
        };
    } // namespace detail

    inline Event whenAll(std::span<Event const> events)
    {
        if(events.empty())
            return readyEvent();
        if(events.size() == 1u)
            return events.front();

        auto state = std::make_shared<detail::WhenAllState>(events.size());
        Event result{state};
        for(auto const& event : events)
            event.subscribe([state, event] { state->arrive(event); });
        return result;
    }

    template<typename T>
    class Future
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);

    public:
        Future() = default;

        bool valid() const noexcept
        {
            return static_cast<bool>(m_state);
        }

        Event event() const
        {
            if(!m_state)
                throw std::logic_error("Invalid Caravan future");
            return Event{m_state};
        }

        CompletionState state() const
        {
            return event().state();
        }

        T const& result() const
        {
            event().wait();
            return m_state->value();
        }

        template<typename T_Executor, typename T_Operation>
        auto then(T_Executor& executor, T_Operation&& operation) const;

    private:
        explicit Future(std::shared_ptr<detail::FutureState<T>> state) : m_state(std::move(state))
        {
        }

        std::shared_ptr<detail::FutureState<T>> m_state;

        friend class Promise<T>;
    };

    template<typename T>
    class Promise
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);

    public:
        Promise() : m_state(std::make_shared<detail::FutureState<T>>())
        {
        }

        Future<T> future() const
        {
            return Future<T>{m_state};
        }

        template<typename U = T>
        bool setValue(U&& value) const
        {
            return m_state->setValue(std::forward<U>(value));
        }

        bool setFailed(std::exception_ptr error) const
        {
            if(!error)
                error = std::make_exception_ptr(std::runtime_error("Caravan operation failed without an error"));
            return m_state->complete(CompletionState::failed, std::move(error));
        }

        bool cancel() const
        {
            return m_state->complete(CompletionState::cancelled);
        }

    private:
        std::shared_ptr<detail::FutureState<T>> m_state;
    };

    template<typename T>
    template<typename T_Executor, typename T_Operation>
    auto Future<T>::then(T_Executor& executor, T_Operation&& operation) const
    {
        using Result = std::invoke_result_t<std::decay_t<T_Operation>&, T const&>;
        auto predecessor = *this;
        auto work = std::make_shared<std::decay_t<T_Operation>>(std::forward<T_Operation>(operation));

        if constexpr(std::is_void_v<Result>)
        {
            EventSource successor;
            auto result = successor.event();
            event().subscribe(
                [predecessor, successor, work, executor = &executor]
                {
                    auto predecessorEvent = predecessor.event();
                    if(predecessorEvent.state() == CompletionState::failed)
                    {
                        successor.setFailed(predecessorEvent.error());
                        return;
                    }
                    if(predecessorEvent.state() == CompletionState::cancelled)
                    {
                        successor.cancel();
                        return;
                    }
                    auto task = [predecessor, successor, work]
                    {
                        ExecutorThreadGuard guard;
                        try
                        {
                            std::invoke(*work, predecessor.result());
                            successor.setReady();
                        }
                        catch(...)
                        {
                            successor.setFailed(std::current_exception());
                        }
                    };
                    try
                    {
                        executor->post(std::move(task));
                    }
                    catch(...)
                    {
                        successor.setFailed(std::current_exception());
                    }
                });
            return result;
        }
        else
        {
            using Value = std::remove_cv_t<std::remove_reference_t<Result>>;
            Promise<Value> successor;
            auto result = successor.future();
            event().subscribe(
                [predecessor, successor, work, executor = &executor]
                {
                    auto predecessorEvent = predecessor.event();
                    if(predecessorEvent.state() == CompletionState::failed)
                    {
                        successor.setFailed(predecessorEvent.error());
                        return;
                    }
                    if(predecessorEvent.state() == CompletionState::cancelled)
                    {
                        successor.cancel();
                        return;
                    }
                    auto task = [predecessor, successor, work]
                    {
                        ExecutorThreadGuard guard;
                        try
                        {
                            successor.setValue(std::invoke(*work, predecessor.result()));
                        }
                        catch(...)
                        {
                            successor.setFailed(std::current_exception());
                        }
                    };
                    try
                    {
                        executor->post(std::move(task));
                    }
                    catch(...)
                    {
                        successor.setFailed(std::current_exception());
                    }
                });
            return result;
        }
    }
} // namespace caravan
