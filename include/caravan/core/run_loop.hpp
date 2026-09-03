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

} // namespace caravan
