/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan
{
    class RunLoopScheduler;
    class RunLoopScheduleSender;

    /** Manually driven single-thread queue for host/control work. */
    class RunLoop
    {
    public:
        RunLoopScheduler scheduler() noexcept;

        void run()
        {
            while(runOne())
            {
            }
        }

        /** Execute one task, blocking until work is ready or the loop is finished. */
        bool runOne()
        {
            ExecutorThreadGuard guard;
            std::function<void()> task;
            {
                std::unique_lock lock(m_mutex);
                m_ready.wait(lock, [this] { return m_finished || !m_tasks.empty(); });
                if(m_tasks.empty())
                    return false;
                task = std::move(m_tasks.front());
                m_tasks.pop_front();
            }
            task();
            return true;
        }

        /** Execute a snapshot of ready work without blocking.
         *
         * Work posted while this batch runs is deferred to the next call, so a
         * self-reposting task cannot monopolize the caller.
         */
        void runReady()
        {
            ExecutorThreadGuard guard;
            std::size_t ready;
            {
                std::lock_guard lock(m_mutex);
                ready = m_tasks.size();
            }
            while(ready-- > 0u)
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
        RunLoopScheduleSender schedule() const noexcept;

    private:
        // Callable submission is only for the scheduling sender and eager wait observers.
        template<typename T_Function>
        void post(T_Function&& function) const
        {
            m_loop->post(std::forward<T_Function>(function));
        }

        explicit RunLoopScheduler(RunLoop& loop) : m_loop(&loop)
        {
        }

        RunLoop* m_loop;

        friend class RunLoop;
        friend class RunLoopScheduleSender;
        friend class Event;
    };

    /** Lazy scheduling operation for the manually driven run loop. */
    class RunLoopScheduleSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit RunLoopScheduleSender(RunLoopScheduler scheduler) : m_scheduler(scheduler)
        {
        }

        template<typename T_Receiver>
        class Operation
        {
        public:
            Operation(RunLoopScheduler scheduler, T_Receiver receiver)
                : m_scheduler(scheduler)
                , m_receiver(std::move(receiver))
            {
            }

            Operation(Operation const&) = delete;
            Operation& operator=(Operation const&) = delete;
            Operation(Operation&&) = delete;
            Operation& operator=(Operation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    m_scheduler.post([this] { m_receiver.set_value(); });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

        private:
            RunLoopScheduler m_scheduler;
            T_Receiver m_receiver;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{m_scheduler, std::forward<T_Receiver>(receiver)};
        }

    private:
        RunLoopScheduler m_scheduler;
    };

    inline RunLoopScheduleSender RunLoopScheduler::schedule() const noexcept
    {
        return RunLoopScheduleSender{*this};
    }

    inline RunLoopScheduler RunLoop::scheduler() noexcept
    {
        return RunLoopScheduler{*this};
    }

} // namespace caravan
