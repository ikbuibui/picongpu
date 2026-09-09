/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan::alpaka
{
    /** Alpaka view plus the allocation handle retained by an operation. */
    template<typename T_View, typename T_Allocation>
    struct OwnedView
    {
        T_View view;
        T_Allocation allocation;
    };

    /** Kernel argument plus an allocation handle retained only for lifetime. */
    template<typename T_Argument, typename T_Allocation>
    struct Retained
    {
        T_Argument argument;
        T_Allocation allocation;
    };

    template<typename T_Argument, typename T_View, typename T_Allocation>
    auto retain(T_Argument argument, OwnedView<T_View, T_Allocation> const& owner)
    {
        return Retained<T_Argument, T_Allocation>{std::move(argument), owner.allocation};
    }

    namespace detail
    {
        template<typename T>
        decltype(auto) nativeHandle(T& value)
        {
            return (value);
        }

        template<typename T_View, typename T_Allocation>
        T_View& nativeHandle(OwnedView<T_View, T_Allocation>& value)
        {
            return value.view;
        }

        template<typename T>
        decltype(auto) nativeArgument(T& value)
        {
            return (value);
        }

        template<typename T_Argument, typename T_Allocation>
        T_Argument& nativeArgument(Retained<T_Argument, T_Allocation>& value)
        {
            return value.argument;
        }

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
                            // ponytail: linear scans at 100 us; tune/back off if measured polling cost warrants it.
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

        template<typename T_Queue, typename T_Receiver, typename... T_Submits>
        class SubmitOperation : private CompletionTask
        {
            static constexpr auto stageCount = sizeof...(T_Submits);
            using Fence = CompletionFence<T_Queue>;

        public:
            SubmitOperation(
                std::array<T_Queue*, stageCount> queues,
                std::tuple<T_Submits...> submits,
                T_Receiver receiver)
                : m_completionThread(completionThread())
                , m_queues(queues)
                , m_submits(std::move(submits))
                , m_receiver(std::move(receiver))
            {
                // Allocate native events before any work can borrow the operation's captures. Same-queue runs
                // share one fence; each queue change supplies both a dependency and an error/lifetime boundary.
                for(std::size_t i = 0u; i < stageCount; ++i)
                    if(i + 1u == stageCount || *m_queues[i] != *m_queues[i + 1u])
                        m_fences[i].emplace(*m_queues[i]);
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
                    m_error = std::current_exception();
                    try
                    {
                        // The throwing submission may already have enqueued work. Fence that queue as well as
                        // earlier runs, without synchronizing here (start may itself run on a progress thread).
                        auto index = m_activeStage;
                        while(!m_fences[index])
                            ++index;
                        m_fences[index]->record(*m_queues[m_activeStage]);
                    }
                    catch(...)
                    {
                        // No fence means no proof that retained storage can be reclaimed.
                        std::terminate();
                    }
                }
                m_completionThread.post(*this);
            }

        private:
            template<std::size_t T_Index>
            void submitStage()
            {
                m_activeStage = T_Index;
                auto& queue = *m_queues[T_Index];
                if constexpr(T_Index > 0u)
                {
                    if(queue != *m_queues[T_Index - 1u])
                        m_fences[T_Index - 1u]->waitOn(queue);
                }

                std::invoke(std::get<T_Index>(m_submits), queue);

                if constexpr(T_Index + 1u < stageCount)
                {
                    if(queue != *m_queues[T_Index + 1u])
                        m_fences[T_Index]->record(queue);
                    submitStage<T_Index + 1u>();
                }
                else
                    m_fences[T_Index]->record(queue);
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
            std::array<std::optional<Fence>, stageCount> m_fences;
            T_Receiver m_receiver;
            std::exception_ptr m_error;
            std::size_t m_activeStage = 0u;
        };
    } // namespace detail

    /** Compatibility query: Caravan terminal completion no longer runs in alpaka callbacks. */
    inline bool isCompletionCallback() noexcept
    {
        return false;
    }

    /** Lazy alpaka-native chain over borrowed caller-supplied queues.
     *
     * Every queue must outlive the connected operation. Submit callables and primitive arguments are retained by value
     * in operation state; any storage referenced by views remains borrowed according to the view's alpaka semantics.
     * Same-queue stages use FIFO. Each queue run ends with a submission-time fence, also used for the next queue's
     * native wait. CPU fences snapshot preceding unobserved task exceptions; other backends use alpaka events.
     * A shared progress thread polls all recorded fences before publishing completion and reclaiming retained state.
     * Submission errors take precedence over execution errors. If a cleanup fence cannot be recorded, or a native
     * event query fails without proving quiescence, Caravan terminates rather than reclaim potentially live storage.
     * Receivers run on an executor thread: use continuesOn before unrestricted or potentially blocking continuations.
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
        friend auto sequence(SubmitSender<T_OtherQueue, T_Left...>, SubmitSender<T_OtherQueue, T_Right...>);

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

    /** Alpaka-domain sequencing preserving FIFO/events instead of crossing host-visible completion. */
    template<typename T_Queue, typename... T_Left, typename... T_Right>
    auto sequence(SubmitSender<T_Queue, T_Left...> left, SubmitSender<T_Queue, T_Right...> right)
    {
        std::array<T_Queue*, sizeof...(T_Left) + sizeof...(T_Right)> queues;
        auto output = queues.begin();
        output = std::copy(left.m_queues.begin(), left.m_queues.end(), output);
        std::copy(right.m_queues.begin(), right.m_queues.end(), output);
        return SubmitSender<T_Queue, T_Left..., T_Right...>{
            queues,
            std::tuple_cat(std::move(left.m_submits), std::move(right.m_submits))};
    }

    /** Pipe adaptor preserving alpaka-native sequencing: previous | sequence(next). */
    template<typename T_Queue, typename... T_Submits>
    auto sequence(SubmitSender<T_Queue, T_Submits...> next)
    {
        return caravan::detail::SenderAdaptorClosure{[next = std::move(next)](auto previous) mutable
                                                     { return sequence(std::move(previous), std::move(next)); }};
    }

    /** Compatibility spelling; generic caravan::then transforms values instead. */
    template<typename T_Queue, typename... T_Left, typename... T_Right>
    [[deprecated("use caravan::alpaka::sequence")]] auto then(
        SubmitSender<T_Queue, T_Left...> left,
        SubmitSender<T_Queue, T_Right...> right)
    {
        return sequence(std::move(left), std::move(right));
    }

    /** Lazy byte fill. The buffer/view and any explicit owner are retained by value. */
    template<typename T_Queue, typename T_Buffer>
    auto fill(T_Queue& queue, T_Buffer buffer, std::uint8_t byte)
    {
        return submit(
            queue,
            [buffer = std::move(buffer), byte](T_Queue& nativeQueue) mutable
            { ::alpaka::memset(nativeQueue, detail::nativeHandle(buffer), byte); });
    }

    /** Lazy copy. Buffer/views, explicit owners, and the extent are retained by value. */
    template<typename T_Queue, typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Queue& queue, T_Destination destination, T_Source source, T_Extent extent)
    {
        return submit(
            queue,
            [destination = std::move(destination), source = std::move(source), extent](T_Queue& nativeQueue) mutable
            {
                ::alpaka::memcpy(nativeQueue, detail::nativeHandle(destination), detail::nativeHandle(source), extent);
            });
    }

    /** Lazy one-element copy for size values. */
    template<typename T_Queue, typename T_Destination, typename T_Source>
    auto size(T_Queue& queue, T_Destination destination, T_Source source)
    {
        using Source = std::remove_cvref_t<decltype(detail::nativeHandle(source))>;
        return copy(
            queue,
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<Source>, ::alpaka::Idx<Source>>::ones());
    }

    /** Lazy kernel launch retaining work division, kernel, arguments, and explicit owners. */
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
                    [&](auto&... values)
                    { ::alpaka::exec<T_Acc>(nativeQueue, workDiv, kernel, detail::nativeArgument(values)...); },
                    args);
            });
    }
} // namespace caravan::alpaka
