/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender.hpp>

namespace caravan::alpaka
{
    namespace detail
    {
        template<typename T_Queue, typename T_Receiver, typename... T_Submits>
        class SubmitOperation
        {
            static constexpr auto stageCount = sizeof...(T_Submits);
            using NativeEvent = ::alpaka::Event<T_Queue>;

        public:
            SubmitOperation(
                std::array<T_Queue*, stageCount> queues,
                std::tuple<T_Submits...> submits,
                T_Receiver receiver)
                : m_queues(queues)
                , m_submits(std::move(submits))
                , m_receiver(std::move(receiver))
            {
            }

            SubmitOperation(SubmitOperation const&) = delete;
            SubmitOperation& operator=(SubmitOperation const&) = delete;
            SubmitOperation(SubmitOperation&&) = delete;
            SubmitOperation& operator=(SubmitOperation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    submitStage<0u>();
                }
                catch(...)
                {
                    auto error = std::current_exception();
                    // Submission may have partially succeeded. Keep captures alive until every borrowed queue is safe.
                    for(auto* queue : m_queues)
                    {
                        try
                        {
                            ::alpaka::wait(*queue);
                        }
                        catch(...)
                        {
                            error = std::current_exception();
                        }
                    }
                    complete([&] { m_receiver.set_error(std::move(error)); });
                }
            }

        private:
            template<std::size_t T_Index>
            void submitStage()
            {
                auto& queue = *m_queues[T_Index];
                if constexpr(T_Index > 0u)
                {
                    if(queue != *m_queues[T_Index - 1u])
                        ::alpaka::wait(queue, *m_events[T_Index - 1u]);
                }

                std::invoke(std::get<T_Index>(m_submits), queue);

                if constexpr(T_Index + 1u < stageCount)
                {
                    if(queue != *m_queues[T_Index + 1u])
                    {
                        m_events[T_Index].emplace(::alpaka::getDev(queue));
                        ::alpaka::enqueue(queue, *m_events[T_Index]);
                    }
                    submitStage<T_Index + 1u>();
                }
                else
                    // This is the CPU policy too: completion is an alpaka queue callback, never a polling thread.
                    ::alpaka::enqueue(queue, [this]() noexcept { complete([this] { m_receiver.set_value(); }); });
            }

            template<typename T_Complete>
            void complete(T_Complete&& complete) noexcept
            {
                if(!m_completed.exchange(true))
                    std::invoke(std::forward<T_Complete>(complete));
            }

            std::array<T_Queue*, stageCount> m_queues;
            std::tuple<T_Submits...> m_submits;
            std::array<std::optional<NativeEvent>, stageCount> m_events;
            T_Receiver m_receiver;
            std::atomic<bool> m_completed = false;
        };
    } // namespace detail

    /** Lazy alpaka-native chain over borrowed caller-supplied queues.
     *
     * Every queue must outlive the connected operation. Submit callables and primitive arguments are retained by value
     * in operation state; any storage referenced by views remains borrowed according to the view's alpaka semantics.
     * Same-queue stages use FIFO. A queue change records an alpaka event and inserts a native queue wait. Only the
     * final queue callback publishes host-visible completion, so intermediate accelerator dependencies never
     * host-wait.
     */
    template<typename T_Queue, typename... T_Submits>
    class SubmitSender
    {
        static constexpr auto stageCount = sizeof...(T_Submits);
        static_assert(stageCount > 0u, "An alpaka submission chain must contain at least one stage");

    public:
        using completion_signatures
            = CompletionSignatures<ValueSignature<>, ErrorSignature<std::exception_ptr>, StoppedSignature>;

        SubmitSender(std::array<T_Queue*, stageCount> queues, std::tuple<T_Submits...> submits)
            : m_queues(queues)
            , m_submits(std::move(submits))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::SubmitOperation<T_Queue, std::decay_t<T_Receiver>, T_Submits...>{
                m_queues,
                std::move(m_submits),
                std::forward<T_Receiver>(receiver)};
        }

        template<typename, typename...>
        friend class SubmitSender;

        template<typename T_OtherQueue, typename... T_Left, typename... T_Right>
        friend auto then(SubmitSender<T_OtherQueue, T_Left...>, SubmitSender<T_OtherQueue, T_Right...>);

    private:
        std::array<T_Queue*, stageCount> m_queues;
        std::tuple<T_Submits...> m_submits;
    };

    /** Lazily describe one native submission stage. The queue is borrowed. */
    template<typename T_Queue, typename T_Submit>
    auto submit(T_Queue& queue, T_Submit submit)
    {
        static_assert(::alpaka::isQueue<T_Queue>);
        using Submit = std::decay_t<T_Submit>;
        return SubmitSender<T_Queue, Submit>{{&queue}, {std::move(submit)}};
    }

    /** Alpaka-domain composition preserving FIFO/events instead of crossing host-visible completion. */
    template<typename T_Queue, typename... T_Left, typename... T_Right>
    auto then(SubmitSender<T_Queue, T_Left...> left, SubmitSender<T_Queue, T_Right...> right)
    {
        std::array<T_Queue*, sizeof...(T_Left) + sizeof...(T_Right)> queues;
        auto output = queues.begin();
        output = std::copy(left.m_queues.begin(), left.m_queues.end(), output);
        std::copy(right.m_queues.begin(), right.m_queues.end(), output);
        return SubmitSender<T_Queue, T_Left..., T_Right...>{
            queues,
            std::tuple_cat(std::move(left.m_submits), std::move(right.m_submits))};
    }

    /** Lazy byte fill. The buffer/view handle is retained by value. */
    template<typename T_Queue, typename T_Buffer>
    auto fill(T_Queue& queue, T_Buffer buffer, std::uint8_t byte)
    {
        return submit(
            queue,
            [buffer = std::move(buffer), byte](T_Queue& nativeQueue) mutable
            { ::alpaka::memset(nativeQueue, buffer, byte); });
    }

    /** Lazy copy. Buffer/view handles and the extent are retained by value. */
    template<typename T_Queue, typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Queue& queue, T_Destination destination, T_Source source, T_Extent extent)
    {
        return submit(
            queue,
            [destination = std::move(destination), source = std::move(source), extent](T_Queue& nativeQueue) mutable
            { ::alpaka::memcpy(nativeQueue, destination, source, extent); });
    }

    /** Lazy one-element copy for PMacc size values. */
    template<typename T_Queue, typename T_Destination, typename T_Source>
    auto size(T_Queue& queue, T_Destination destination, T_Source source)
    {
        using Source = std::remove_cvref_t<T_Source>;
        return copy(
            queue,
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<Source>, ::alpaka::Idx<Source>>::ones());
    }

    /** Lazy kernel launch. Work division, kernel and arguments are retained by value. */
    template<typename T_Acc, typename T_Queue, typename T_WorkDiv, typename T_Kernel, typename... T_Args>
    auto kernel(T_Queue& queue, T_WorkDiv workDiv, T_Kernel kernel, T_Args... args)
    {
        return submit(
            queue,
            [workDiv = std::move(workDiv),
             kernel = std::move(kernel),
             args = std::tuple<T_Args...>{std::move(args)...}](T_Queue& nativeQueue) mutable
            {
                std::apply(
                    [&](auto&... values) { ::alpaka::exec<T_Acc>(nativeQueue, workDiv, kernel, values...); },
                    args);
            });
    }
} // namespace caravan::alpaka
