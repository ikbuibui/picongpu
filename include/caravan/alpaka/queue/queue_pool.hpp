/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/alpaka/submission.hpp>

namespace caravan::alpaka
{
    template<typename T_Pool>
    struct PoolSubmissionDomain;

    template<typename T_Pool, typename... T_Submits>
    class PoolSubmitSender;

    template<typename T_Queue>
    class QueuePool;

    template<typename T_Queue>
    using PooledSubmissionDomain = PoolSubmissionDomain<QueuePool<T_Queue>>;

    template<typename T_Queue, typename... T_Submits>
    using PooledSubmitSender = PoolSubmitSender<QueuePool<T_Queue>, T_Submits...>;

    namespace detail
    {
        template<typename T_Pool, typename T_Receiver, typename... T_Submits>
        class PooledSubmitOperation;
    } // namespace detail

    /** Device-local queue pool for automatically placed submission graphs.
     *
     * submissions() returns a lightweight pool-bound factory. Sender construction remains lazy; connecting a
     * composed graph leases all of its logical queues together, growing the pool rather than waiting. A lease is
     * returned after native completion, or by destruction if its operation was never started. The pool must outlive
     * its senders and connected operations.
     */
    template<typename T_Queue>
    class QueuePool
    {
        static_assert(::alpaka::isQueue<T_Queue>);

        struct Entry
        {
            explicit Entry(::alpaka::Dev<T_Queue> const& device) : queue(device)
            {
            }

            T_Queue queue;
            bool leased = false;
        };

    public:
        using Queue = T_Queue;

        class Lease
        {
        public:
            Lease(Lease const&) = delete;
            Lease& operator=(Lease const&) = delete;

            Lease(Lease&& other) noexcept
                : m_pool(std::exchange(other.m_pool, nullptr))
                , m_entries(std::move(other.m_entries))
            {
            }

            ~Lease()
            {
                if(m_pool)
                    m_pool->release(m_entries);
            }

            T_Queue& operator[](std::size_t index) const noexcept
            {
                return m_entries[index]->queue;
            }

            template<typename T_Operation>
            void start(T_Operation& operation) const noexcept
            {
                operation.start();
            }

        private:
            friend class QueuePool;

            Lease(QueuePool& pool, std::vector<Entry*> entries) : m_pool(&pool), m_entries(std::move(entries))
            {
            }

            QueuePool* m_pool;
            std::vector<Entry*> m_entries;
        };

        using Binding = Lease;

        class SubmissionFactory
        {
        public:
            template<typename T_Submit>
            auto submit(T_Submit submit) const;

        private:
            friend class QueuePool;

            explicit SubmissionFactory(QueuePool& pool) : m_pool(&pool)
            {
            }

            QueuePool* m_pool;
        };

        explicit QueuePool(::alpaka::Dev<T_Queue> device) : m_device(std::move(device))
        {
        }

        QueuePool(QueuePool const&) = delete;
        QueuePool& operator=(QueuePool const&) = delete;

        SubmissionFactory submissions() noexcept
        {
            return SubmissionFactory{*this};
        }

    private:
        template<typename, typename, typename...>
        friend class detail::PooledSubmitOperation;

        Lease acquire(std::size_t count)
        {
            std::lock_guard lock(m_mutex);
            std::vector<Entry*> entries;
            entries.reserve(count);
            for(auto const& entry : m_entries)
                if(!entry->leased && entries.size() != count)
                    entries.push_back(entry.get());
            // ponytail: grow instead of blocking; add an explicit cap only if measured resource use requires it.
            while(entries.size() != count)
            {
                auto entry = std::make_unique<Entry>(m_device);
                entries.push_back(entry.get());
                m_entries.push_back(std::move(entry));
            }
            for(auto* entry : entries)
                entry->leased = true;
            return Lease{*this, std::move(entries)};
        }

        void release(std::vector<Entry*> const& entries) noexcept
        {
            std::lock_guard lock(m_mutex);
            for(auto* entry : entries)
                entry->leased = false;
        }

        ::alpaka::Dev<T_Queue> m_device;
        std::mutex m_mutex;
        std::vector<std::unique_ptr<Entry>> m_entries;
    };

    namespace detail
    {
        template<typename T_Pool, typename T_Receiver, typename... T_Submits>
        class PooledSubmitOperation
        {
            static constexpr auto stageCount = sizeof...(T_Submits);
            using T_Queue = typename T_Pool::Queue;
            using Binding = typename T_Pool::Binding;

            struct Receiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    binding->reset();
                    receiver.set_value(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    binding->reset();
                    receiver.set_error(std::move(error));
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& output) { output.get_env(); }
                {
                    return receiver.get_env();
                }

                std::optional<Binding>* binding;
                T_Receiver receiver;
            };

            using Operation = SubmitOperation<T_Queue, Receiver, T_Submits...>;

            static auto bindQueues(Binding const& binding, std::array<std::size_t, stageCount> const& lanes)
            {
                std::array<T_Queue*, stageCount> queues;
                for(std::size_t i = 0u; i < stageCount; ++i)
                    queues[i] = &binding[lanes[i]];
                return queues;
            }

        public:
            PooledSubmitOperation(
                T_Pool& pool,
                std::size_t laneCount,
                std::array<std::size_t, stageCount> lanes,
                std::tuple<T_Submits...> submits,
                SubmissionDependencies<stageCount> dependencies,
                T_Receiver receiver)
                : m_binding(std::in_place, pool.acquire(laneCount))
                , m_operation(
                      bindQueues(*m_binding, lanes),
                      std::move(submits),
                      dependencies,
                      Receiver{&m_binding, std::move(receiver)})
            {
            }

            PooledSubmitOperation(PooledSubmitOperation const&) = delete;
            PooledSubmitOperation& operator=(PooledSubmitOperation const&) = delete;
            PooledSubmitOperation(PooledSubmitOperation&&) = delete;
            PooledSubmitOperation& operator=(PooledSubmitOperation&&) = delete;

            void start() & noexcept
            {
                m_binding->start(m_operation);
            }

        private:
            std::optional<Binding> m_binding;
            Operation m_operation;
        };
    } // namespace detail

    /** Lazy alpaka submissions whose queues are assigned by a pool at connection.
     *
     * Native sequence and whenAll composition requires one pool. Use caravan::genericWhenAll to join submissions
     * from different pools after their independent host-visible completions.
     */
    template<typename T_Pool, typename... T_Submits>
    class PoolSubmitSender
    {
        static constexpr auto stageCount = sizeof...(T_Submits);
        static_assert(stageCount > 0u, "An alpaka submission chain must contain at least one stage");

    public:
        using completion_signatures = CompletionSignatures<ValueSignature<>, ErrorSignature<std::exception_ptr>>;

        PoolSubmitSender(
            T_Pool& pool,
            std::tuple<T_Submits...> submits,
            std::array<std::size_t, stageCount> lanes = {},
            std::size_t laneCount = 1u,
            detail::SubmissionDependencies<stageCount> dependencies
            = detail::SubmissionDependencies<stageCount>::linear())
            : m_pool(&pool)
            , m_submits(std::move(submits))
            , m_lanes(lanes)
            , m_laneCount(laneCount)
            , m_dependencies(dependencies)
        {
        }

        auto query(GetDomain) const noexcept -> PoolSubmissionDomain<T_Pool>
        {
            return {};
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::PooledSubmitOperation<T_Pool, std::decay_t<T_Receiver>, T_Submits...>{
                *m_pool,
                m_laneCount,
                m_lanes,
                std::move(m_submits),
                m_dependencies,
                std::forward<T_Receiver>(receiver)};
        }

        template<typename, typename...>
        friend class PoolSubmitSender;

        friend struct PoolSubmissionDomain<T_Pool>;

        template<typename T_OtherPool, typename... T_Left, typename... T_Right>
        friend auto sequence(PoolSubmitSender<T_OtherPool, T_Left...>, PoolSubmitSender<T_OtherPool, T_Right...>);

    private:
        template<bool T_Ordered, typename... T_Right>
        auto compose(PoolSubmitSender<T_Pool, T_Right...> right) &&
        {
            if(m_pool != right.m_pool)
                throw std::invalid_argument(
                    "Native alpaka composition requires one queue pool; use caravan::letValue for sequencing "
                    "or caravan::genericWhenAll for joining independent branches");

            constexpr auto rightCount = sizeof...(T_Right);
            std::array<std::size_t, stageCount + rightCount> lanes;
            auto output = std::copy(m_lanes.begin(), m_lanes.end(), lanes.begin());
            if constexpr(T_Ordered)
                std::copy(right.m_lanes.begin(), right.m_lanes.end(), output);
            else
                std::transform(
                    right.m_lanes.begin(),
                    right.m_lanes.end(),
                    output,
                    [offset = m_laneCount](std::size_t lane) { return offset + lane; });

            auto laneCount = T_Ordered ? std::max(m_laneCount, right.m_laneCount) : m_laneCount + right.m_laneCount;
            return PoolSubmitSender<T_Pool, T_Submits..., T_Right...>{
                *m_pool,
                std::tuple_cat(std::move(m_submits), std::move(right.m_submits)),
                lanes,
                laneCount,
                detail::composeDependencies<T_Ordered>(m_dependencies, right.m_dependencies)};
        }

        T_Pool* m_pool;
        std::tuple<T_Submits...> m_submits;
        std::array<std::size_t, stageCount> m_lanes;
        std::size_t m_laneCount;
        detail::SubmissionDependencies<stageCount> m_dependencies;
    };

    /** Native lowering for pool-bound submissions; each whenAll child receives disjoint logical lanes. */
    template<typename T_Pool>
    struct PoolSubmissionDomain
    {
        template<typename... T_Left, typename... T_Right>
        auto transform(
            SequenceTag,
            PoolSubmitSender<T_Pool, T_Left...> left,
            PoolSubmitSender<T_Pool, T_Right...> right) const
        {
            return std::move(left).template compose<true>(std::move(right));
        }

        template<typename... T_Submits>
        auto transform(WhenAllTag, PoolSubmitSender<T_Pool, T_Submits...> sender) const
        {
            return sender;
        }

        template<typename... T_Left, typename... T_Right, typename... T_Rest>
        auto transform(
            WhenAllTag tag,
            PoolSubmitSender<T_Pool, T_Left...> left,
            PoolSubmitSender<T_Pool, T_Right...> right,
            T_Rest... rest) const
        {
            return transform(tag, std::move(left).template compose<false>(std::move(right)), std::move(rest)...);
        }
    };

    template<typename T_Queue>
    template<typename T_Submit>
    auto QueuePool<T_Queue>::SubmissionFactory::submit(T_Submit submit) const
    {
        using Submit = std::decay_t<T_Submit>;
        return PooledSubmitSender<T_Queue, Submit>{*m_pool, {std::move(submit)}};
    }

    /** Alpaka-native sequencing with automatic queue affinity from a shared pool. */
    template<typename T_Pool, typename... T_Left, typename... T_Right>
    auto sequence(PoolSubmitSender<T_Pool, T_Left...> left, PoolSubmitSender<T_Pool, T_Right...> right)
    {
        return std::move(left).template compose<true>(std::move(right));
    }

    template<typename T_Pool, typename... T_Submits>
    auto sequence(PoolSubmitSender<T_Pool, T_Submits...> next)
    {
        return caravan::detail::SenderAdaptorClosure{
            [next = std::move(next)](auto previous) mutable
            { return caravan::alpaka::sequence(std::move(previous), std::move(next)); }};
    }
} // namespace caravan::alpaka
