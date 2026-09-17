/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <caravan/core/eager.hpp>

namespace caravan
{
    /** Fixed-size worker pool for host submission continuations.
     *
     * Used to move device-graph submission work off the single MPI owner thread so independent halo
     * directions can be submitted concurrently. Tasks are ordinary host callables and must not block
     * MPI progress; the pool is intentionally separate from the MPI context.
     */
    class ThreadPool
    {
    public:
        explicit ThreadPool(std::size_t workerCount)
        {
            m_workers.reserve(workerCount);
            for(std::size_t i = 0u; i < workerCount; ++i)
                m_workers.emplace_back([this] { run(); });
        }

        ~ThreadPool()
        {
            {
                std::lock_guard lock(m_mutex);
                m_stopping = true;
            }
            m_ready.notify_all();
            for(auto& worker : m_workers)
                worker.join();
        }

        ThreadPool(ThreadPool const&) = delete;
        ThreadPool& operator=(ThreadPool const&) = delete;

        void post(std::function<void()> task)
        {
            {
                std::lock_guard lock(m_mutex);
                m_tasks.push_back(std::move(task));
            }
            m_ready.notify_one();
        }

    private:
        void run()
        {
            ExecutorThreadGuard guard;
            while(true)
            {
                std::function<void()> task;
                {
                    std::unique_lock lock(m_mutex);
                    m_ready.wait(lock, [this] { return m_stopping || !m_tasks.empty(); });
                    if(m_tasks.empty())
                        return;
                    task = std::move(m_tasks.front());
                    m_tasks.pop_front();
                }
                task();
            }
        }

        std::mutex m_mutex;
        std::condition_variable m_ready;
        std::deque<std::function<void()>> m_tasks;
        std::vector<std::thread> m_workers;
        bool m_stopping = false;
    };

    /** Number of submission workers. Zero (the default) keeps every continuation inline. */
    inline std::size_t submissionThreadCount() noexcept
    {
        auto const* value = std::getenv("CARAVAN_SUBMISSION_THREADS");
        if(value == nullptr || *value == '\0')
            return 0u;
        return static_cast<std::size_t>(std::strtoul(value, nullptr, 10));
    }

    /** Process-wide submission executor; inline when no workers were requested. */
    class SubmissionExecutor
    {
    public:
        static SubmissionExecutor& instance()
        {
            // Leaked intentionally: senders may outlive other static destructors, and the worker
            // threads are safe to leave blocked at process exit.
            static SubmissionExecutor* executor = new SubmissionExecutor();
            return *executor;
        }

        void post(std::function<void()> task)
        {
            if(m_pool)
                m_pool->post(std::move(task));
            else
                task();
        }

        std::size_t threadCount() const noexcept
        {
            return m_pool ? submissionThreadCount() : 0u;
        }

    private:
        SubmissionExecutor()
        {
            if(auto const count = submissionThreadCount(); count != 0u)
                m_pool = std::make_unique<ThreadPool>(count);
        }

        std::unique_ptr<ThreadPool> m_pool;
    };

    inline SubmissionExecutor& submissionExecutor()
    {
        return SubmissionExecutor::instance();
    }

    /** Scheduler that runs a continuation on the process-wide submission executor. */
    class SubmissionScheduler
    {
        class ScheduleSender
        {
        public:
            using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

            template<typename T_Receiver>
            class Operation
            {
            public:
                explicit Operation(T_Receiver receiver) : m_receiver(std::move(receiver))
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    submissionExecutor().post([this] { m_receiver.set_value(); });
                }

            private:
                T_Receiver m_receiver;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{std::forward<T_Receiver>(receiver)};
            }
        };

    public:
        auto schedule() const noexcept
        {
            return ScheduleSender{};
        }
    };

} // namespace caravan
