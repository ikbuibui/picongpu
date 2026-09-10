/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan::alpaka::detail
{
    /** A submission-time fence. Query failure does not establish quiescence. */
    template<typename T_Queue>
    class CompletionFence
    {
    public:
        explicit CompletionFence(T_Queue const& queue) : m_event(::alpaka::getDev(queue))
        {
        }

        void record(T_Queue& queue)
        {
            if(!m_recorded)
            {
                ::alpaka::enqueue(queue, m_event);
                m_recorded = true;
            }
        }

        void waitOn(T_Queue& queue)
        {
            ::alpaka::wait(queue, m_event);
        }

        bool poll(std::exception_ptr&) noexcept
        {
            if(!m_recorded || m_complete)
                return true;
            try
            {
                m_complete = ::alpaka::isComplete(m_event);
                return m_complete;
            }
            catch(...)
            {
                // An unsuccessful query is not a lifetime fence, even if it reports a device execution error.
                std::terminate();
            }
        }

    private:
        ::alpaka::Event<T_Queue> m_event;
        bool m_recorded = false;
        bool m_complete = false;
    };

    /** CPU barriers snapshot preceding task errors and signal only after those tasks have been destroyed. */
    template<typename T_Dev>
    class CompletionFence<::alpaka::QueueGenericThreadsNonBlocking<T_Dev>>
    {
        using Queue = ::alpaka::QueueGenericThreadsNonBlocking<T_Dev>;

    public:
        explicit CompletionFence(Queue const&)
        {
        }

        void record(Queue& queue)
        {
            if(!m_future.valid())
                m_future = queue.m_spQueueImpl->m_workerThread.submitErrorBarrier().share();
        }

        void waitOn(Queue& queue)
        {
            // Match alpaka's CPU queue-event wait without consuming or losing the fence's error snapshot.
            queue.m_spQueueImpl->m_workerThread.submit([future = m_future] { future.wait(); });
        }

        bool poll(std::exception_ptr& error) noexcept
        {
            if(!m_future.valid() || m_complete)
                return true;
            if(m_future.wait_for(std::chrono::seconds{0}) != std::future_status::ready)
                return false;
            m_complete = true;
            try
            {
                m_future.get();
            }
            catch(...)
            {
                // Unlike fence construction/query failure, this exception is delivered by a completed barrier.
                if(!error)
                    error = std::current_exception();
            }
            return true;
        }

    private:
        std::shared_future<void> m_future;
        bool m_complete = false;
    };

    class CompletionTask
    {
    public:
        // A true result may destroy this task; false retains it for the next scan.
        virtual bool poll() noexcept = 0;

        CompletionTask* next = nullptr;

    protected:
        ~CompletionTask() = default;
    };

    /** Observes terminal fences and delivers receivers without blocking on pending backend work. */
    class CompletionThread
    {
    public:
        CompletionThread() : m_thread([this] { run(); })
        {
        }

        CompletionThread(CompletionThread const&) = delete;
        CompletionThread& operator=(CompletionThread const&) = delete;

        ~CompletionThread()
        {
            {
                std::lock_guard lock(m_mutex);
                m_stopped = true;
            }
            m_ready.notify_one();
            m_thread.join();
        }

        void post(CompletionTask& task) noexcept
        {
            {
                std::lock_guard lock(m_mutex);
                if(m_tail)
                    m_tail->next = &task;
                else
                    m_head = &task;
                m_tail = &task;
            }
            m_ready.notify_one();
        }

    private:
        void run() noexcept
        {
            ExecutorThreadGuard guard;
            CompletionTask* pending = nullptr;
            while(true)
            {
                {
                    std::unique_lock lock(m_mutex);
                    if(pending)
                        // linear scans at 100 us; tune/back off if measured polling cost warrants it.
                        m_ready.wait_for(lock, std::chrono::microseconds{100}, [this] { return m_head; });
                    else
                        m_ready.wait(lock, [this] { return m_stopped || m_head; });
                    if(m_head)
                    {
                        m_tail->next = pending;
                        pending = std::exchange(m_head, nullptr);
                        m_tail = nullptr;
                    }
                    if(!pending && m_stopped)
                        return;
                }

                CompletionTask* deferred = nullptr;
                auto** tail = &deferred;
                while(pending)
                {
                    auto* task = pending;
                    pending = std::exchange(task->next, nullptr);
                    if(!task->poll())
                    {
                        *tail = task;
                        tail = &task->next;
                    }
                }
                pending = deferred;
            }
        }

        std::mutex m_mutex;
        std::condition_variable m_ready;
        CompletionTask* m_head = nullptr;
        CompletionTask* m_tail = nullptr;
        bool m_stopped = false;
        std::thread m_thread;
    };

    inline CompletionThread& completionThread()
    {
        static CompletionThread thread;
        return thread;
    }
} // namespace caravan::alpaka::detail
