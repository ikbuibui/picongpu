/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/alpaka/completion.hpp>
#include <caravan/core/eager.hpp>

namespace caravan::alpaka
{
    template<typename T_Queue>
    struct SubmissionDomain;

    namespace detail
    {
        template<std::size_t T_Count>
        struct SubmissionDependencies
        {
            // O(N^2) storage/scans for fixed submission expressions; use sparse edges if large graphs matter.
            std::array<std::array<bool, T_Count>, T_Count> predecessors{};

            static auto linear()
            {
                SubmissionDependencies result;
                for(std::size_t i = 1u; i < T_Count; ++i)
                    result.predecessors[i][i - 1u] = true;
                return result;
            }
        };

        template<typename T_Queue, typename T_Receiver, typename... T_Submits>
        class SubmitOperation : private CompletionTask
        {
            static constexpr auto stageCount = sizeof...(T_Submits);
            using Fence = CompletionFence<T_Queue>;

        public:
            SubmitOperation(
                std::array<T_Queue*, stageCount> queues,
                std::tuple<T_Submits...> submits,
                SubmissionDependencies<stageCount> dependencies,
                T_Receiver receiver)
                : m_completionThread(completionThread())
                , m_queues(queues)
                , m_submits(std::move(submits))
                , m_dependencies(dependencies)
                , m_receiver(std::move(receiver))
            {
                std::array<std::size_t, stageCount> incoming{}, outgoing{};
                for(std::size_t i = 0u; i < stageCount; ++i)
                    for(std::size_t j = 0u; j < i; ++j)
                        if(m_dependencies.predecessors[i][j])
                        {
                            ++incoming[i];
                            ++outgoing[j];
                        }

                // Preallocate fences before borrowing captures. Only unbranched, adjacent same-queue stages
                // share a fence: after a submission failure the remainder of that run is necessarily skipped.
                // whenAll(A, B) -> C on A's queue retains A's branch fence despite FIFO ordering.
                // Future optimization: omit it when no cross-queue consumer needs it, preserving error snapshots
                // and a cleanup fence for A if submission failure skips C and its terminal fence.
                for(std::size_t i = stageCount; i-- > 0u;)
                {
                    if(i + 1u < stageCount && *m_queues[i] == *m_queues[i + 1u]
                       && m_dependencies.predecessors[i + 1u][i] && outgoing[i] == 1u && incoming[i + 1u] == 1u)
                        m_fenceIndices[i] = m_fenceIndices[i + 1u];
                    else
                    {
                        m_fenceIndices[i] = i;
                        m_fences[i].emplace(*m_queues[i]);
                    }
                }
            }

            SubmitOperation(SubmitOperation const&) = delete;
            SubmitOperation& operator=(SubmitOperation const&) = delete;
            SubmitOperation(SubmitOperation&&) = delete;
            SubmitOperation& operator=(SubmitOperation&&) = delete;

            void start() & noexcept
            {
                submitStage<0u>();
                m_completionThread.post(*this);
            }

        private:
            template<std::size_t T_Index>
            void submitStage() noexcept
            {
                auto& queue = *m_queues[T_Index];
                auto& fence = m_fences[m_fenceIndices[T_Index]];
                for(std::size_t i = 0u; i != T_Index; ++i)
                    if(m_dependencies.predecessors[T_Index][i] && m_failed[i])
                        m_failed[T_Index] = true;

                if(!m_failed[T_Index])
                {
                    try
                    {
                        for(std::size_t i = 0u; i != T_Index; ++i)
                            if(m_dependencies.predecessors[T_Index][i] && queue != *m_queues[i])
                                m_fences[m_fenceIndices[i]]->waitOn(queue);
                        std::invoke(std::get<T_Index>(m_submits), queue);
                        if(m_fenceIndices[T_Index] == T_Index)
                            fence->record(queue);
                    }
                    catch(...)
                    {
                        m_failed[T_Index] = true;
                        if(!m_error)
                            m_error = std::current_exception();
                        try
                        {
                            // A throwing wait/submission may already have borrowed captures. Fence it before
                            // continuing independent branches; descendants of this failed stage are skipped.
                            fence->record(queue);
                        }
                        catch(...)
                        {
                            std::terminate(); // No fence means no proof that retained storage can be reclaimed.
                        }
                    }
                }

                if constexpr(T_Index + 1u < stageCount)
                    submitStage<T_Index + 1u>();
            }

            bool poll() noexcept override
            {
                bool ready = true;
                for(auto& fence : m_fences)
                    if(fence && !fence->poll(m_error))
                        ready = false;
                if(!ready)
                    return false;
                if(m_error)
                    m_receiver.set_error(std::move(m_error));
                else
                    m_receiver.set_value();
                // The receiver may destroy this operation. Do not access members below this point.
                return true;
            }

            CompletionThread& m_completionThread;
            std::array<T_Queue*, stageCount> m_queues;
            std::tuple<T_Submits...> m_submits;
            SubmissionDependencies<stageCount> m_dependencies;
            std::array<std::optional<Fence>, stageCount> m_fences;
            std::array<std::size_t, stageCount> m_fenceIndices{};
            std::array<bool, stageCount> m_failed{};
            T_Receiver m_receiver;
            std::exception_ptr m_error;
        };
    } // namespace detail

    /** Compatibility query: Caravan terminal completion no longer runs in alpaka callbacks. */
    inline bool isCompletionCallback() noexcept
    {
        return false;
    }

    /** Lazy alpaka-native submissions over borrowed caller-supplied queues.
     *
     * Every queue must outlive the connected operation. Captures are retained until all submitted work is quiescent;
     * storage referenced by unowned views remains borrowed. Callables run on the submitting host thread and must
     * only enqueue tracked work on their supplied queue, not read unfinished results or block for completion.
     * sequence uses FIFO/events; whenAll preserves independent branches. Ordinary then/letValue
     * callbacks and explicit placement wrappers remain host-completion boundaries. Unbranched same-queue runs
     * share a fence. CPU fences snapshot task exceptions; other backends use alpaka events.
     * A shared progress thread polls all recorded fences before terminal completion and reclamation. Submission
     * failures skip descendants but not independent branches; asynchronous errors cannot retract queued work.
     * Submission errors take precedence over execution errors. Failure to establish quiescence terminates rather
     * than reclaim live storage. Receivers run on an executor thread: use continuesOn before blocking callbacks.
     */
    template<typename T_Queue, typename... T_Submits>
    class SubmitSender
    {
        static constexpr auto stageCount = sizeof...(T_Submits);
        static_assert(stageCount > 0u, "An alpaka submission chain must contain at least one stage");

    public:
        using completion_signatures
            = CompletionSignatures<ValueSignature<>, ErrorSignature<std::exception_ptr>, StoppedSignature>;

        SubmitSender(
            std::array<T_Queue*, stageCount> queues,
            std::tuple<T_Submits...> submits,
            detail::SubmissionDependencies<stageCount> dependencies
            = detail::SubmissionDependencies<stageCount>::linear())
            : m_queues(queues)
            , m_submits(std::move(submits))
            , m_dependencies(dependencies)
        {
        }

        auto query(GetDomain) const noexcept -> SubmissionDomain<T_Queue>
        {
            return {};
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::SubmitOperation<T_Queue, std::decay_t<T_Receiver>, T_Submits...>{
                m_queues,
                std::move(m_submits),
                m_dependencies,
                std::forward<T_Receiver>(receiver)};
        }

        template<typename, typename...>
        friend class SubmitSender;

        friend struct SubmissionDomain<T_Queue>;

        template<typename T_OtherQueue, typename... T_Left, typename... T_Right>
        friend auto sequence(SubmitSender<T_OtherQueue, T_Left...>, SubmitSender<T_OtherQueue, T_Right...>);

    private:
        template<bool T_Ordered, typename... T_Right>
        auto compose(SubmitSender<T_Queue, T_Right...> right) &&
        {
            constexpr auto rightCount = sizeof...(T_Right);
            std::array<T_Queue*, stageCount + rightCount> queues;
            auto output = std::copy(m_queues.begin(), m_queues.end(), queues.begin());
            std::copy(right.m_queues.begin(), right.m_queues.end(), output);
            detail::SubmissionDependencies<stageCount + rightCount> dependencies;
            std::array<bool, stageCount> tails;
            tails.fill(true);
            for(std::size_t i = 0u; i < stageCount; ++i)
                for(std::size_t j = 0u; j < stageCount; ++j)
                {
                    auto edge = m_dependencies.predecessors[i][j];
                    dependencies.predecessors[i][j] = edge;
                    if(edge)
                        tails[j] = false;
                }
            for(std::size_t i = 0u; i < rightCount; ++i)
            {
                auto const& predecessors = right.m_dependencies.predecessors[i];
                std::copy(
                    predecessors.begin(),
                    predecessors.end(),
                    dependencies.predecessors[stageCount + i].begin() + stageCount);
                if constexpr(T_Ordered)
                    if(std::none_of(predecessors.begin(), predecessors.end(), [](bool edge) { return edge; }))
                        std::copy(tails.begin(), tails.end(), dependencies.predecessors[stageCount + i].begin());
            }
            return SubmitSender<T_Queue, T_Submits..., T_Right...>{
                queues,
                std::tuple_cat(std::move(m_submits), std::move(right.m_submits)),
                dependencies};
        }

        std::array<T_Queue*, stageCount> m_queues;
        std::tuple<T_Submits...> m_submits;
        detail::SubmissionDependencies<stageCount> m_dependencies;
    };

    /** Native lowering only for explicit submissions. Queue type compatibility is not queue identity. */
    template<typename T_Queue>
    struct SubmissionDomain
    {
        template<typename... T_Submits>
        auto transform(WhenAllTag, SubmitSender<T_Queue, T_Submits...> sender) const
        {
            return sender;
        }

        template<typename... T_Left, typename... T_Right, typename... T_Rest>
        auto transform(
            WhenAllTag tag,
            SubmitSender<T_Queue, T_Left...> left,
            SubmitSender<T_Queue, T_Right...> right,
            T_Rest... rest) const
        {
            return transform(tag, std::move(left).template compose<false>(std::move(right)), std::move(rest)...);
        }
    };

    /** Lazily describe one native submission stage. The queue is borrowed. */
    template<typename T_Queue, typename T_Submit>
    auto submit(T_Queue& queue, T_Submit submit)
    {
        static_assert(::alpaka::isQueue<T_Queue>);
        using Submit = std::decay_t<T_Submit>;
        return SubmitSender<T_Queue, Submit>{{&queue}, {std::move(submit)}};
    }

    /** Alpaka-native sequencing preserving FIFO/events instead of crossing host-visible completion. */
    template<typename T_Queue, typename... T_Left, typename... T_Right>
    auto sequence(SubmitSender<T_Queue, T_Left...> left, SubmitSender<T_Queue, T_Right...> right)
    {
        return std::move(left).template compose<true>(std::move(right));
    }

    /** Pipe adaptor preserving alpaka-native sequencing: previous | sequence(next). */
    template<typename T_Queue, typename... T_Submits>
    auto sequence(SubmitSender<T_Queue, T_Submits...> next)
    {
        return caravan::detail::SenderAdaptorClosure{[next = std::move(next)](auto previous) mutable
                                                     { return sequence(std::move(previous), std::move(next)); }};
    }
} // namespace caravan::alpaka
