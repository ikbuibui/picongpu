/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <caravan/alpaka/queue/queue_pool.hpp>

namespace caravan::alpaka
{
    /** Device-local pool shared by all connected graphs.
     *
     * The pool may grow before its first connection and is fixed-size afterwards.
     * At connection, logical lane i maps to (base + i) % queueCount; base advances by the graph's lane count.
     * sequence retains logical affinity, while whenAll branches may serialize on the same physical queue.
     * There are no exclusive leases or completion-time admission waits. The cap applies to this pool instance.
     *
     * Entire graph submissions (including error fences) are serialized to keep CPU queue errors attributable
     * to their graph. Nonblocking queues can execute different graphs concurrently after submission. Callables
     * must only enqueue work on the supplied queue: no recursive graph starts, external queue submissions, or
     * blocking on other branches. Queue references must not escape for later use. The pool must outlive all its
     * senders and connected operations. Cross-pool native composition is rejected, as for QueuePool.
     */
    template<typename T_Queue>
    class SharedQueuePool
    {
        static_assert(::alpaka::isQueue<T_Queue>);

    public:
        using Queue = T_Queue;

        class Binding
        {
        public:
            T_Queue& operator[](std::size_t lane) const noexcept
            {
                return m_pool->m_queues[(m_base + lane) % m_pool->m_queues.size()];
            }

            template<typename T_Operation>
            void start(T_Operation& operation) const noexcept
            {
                operation.start(&m_pool->m_mutex);
            }

        private:
            friend class SharedQueuePool;

            Binding(SharedQueuePool& pool, std::size_t base) : m_pool(&pool), m_base(base)
            {
            }

            SharedQueuePool* m_pool;
            std::size_t m_base;
        };

        class SubmissionFactory
        {
        public:
            template<typename T_Submit>
            auto submit(T_Submit submit) const
            {
                return PoolSubmitSender<SharedQueuePool, std::decay_t<T_Submit>>{*m_pool, {std::move(submit)}};
            }

        private:
            friend class SharedQueuePool;

            explicit SubmissionFactory(SharedQueuePool& pool) : m_pool(&pool)
            {
            }

            SharedQueuePool* m_pool;
        };

        SharedQueuePool(::alpaka::Dev<T_Queue> const& device, std::size_t queueCount)
        {
            if(queueCount == 0u)
                throw std::invalid_argument("SharedQueuePool requires at least one queue");
            m_queues.reserve(queueCount);
            for(std::size_t i = 0u; i < queueCount; ++i)
                m_queues.emplace_back(device);
        }

        SharedQueuePool(SharedQueuePool const&) = delete;
        SharedQueuePool& operator=(SharedQueuePool const&) = delete;

        SubmissionFactory submissions() noexcept
        {
            return SubmissionFactory{*this};
        }

        /** Add queues before the pool is first used. */
        void addQueues(std::size_t count)
        {
            std::lock_guard lock(m_mutex);
            if(m_started)
                throw std::logic_error("SharedQueuePool cannot grow after its first submission");
            while(count-- != 0u)
                m_queues.emplace_back(::alpaka::getDev(m_queues.front()));
        }

        std::size_t size() const
        {
            std::lock_guard lock(m_mutex);
            return m_queues.size();
        }

    private:
        template<typename, typename, typename...>
        friend class detail::PooledSubmitOperation;

        Binding acquire(std::size_t laneCount)
        {
            std::lock_guard lock(m_mutex);
            m_started = true;
            auto base = m_next;
            m_next = (m_next + laneCount % m_queues.size()) % m_queues.size();
            return Binding{*this, base};
        }

        std::vector<T_Queue> m_queues;
        // ponytail: serialize graph submission, not execution; use per-queue fence-run locking if contention matters.
        mutable std::mutex m_mutex;
        std::size_t m_next = 0u;
        bool m_started = false;
    };
} // namespace caravan::alpaka
