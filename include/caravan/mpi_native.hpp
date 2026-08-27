/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <algorithm>
#include <climits>
#include <exception>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/mpi.hpp>
#include <mpi.h>

namespace caravan
{
    class NativeMpiContext;

    namespace detail
    {
        struct NativeAccess;
    } // namespace detail

    /** Native requests and lifetime tokens transferred to the MPI executor.
     *
     * Build the batch before starting requests. Until Caravan accepts the
     * returned batch, its destructor drains requests during exception cleanup.
     */
    class NativeRequestBatch
    {
    public:
        NativeRequestBatch() = default;

        NativeRequestBatch(
            std::vector<MPI_Request> nativeRequests,
            std::vector<std::shared_ptr<void>> lifetimeTokens = {})
            : requests(std::move(nativeRequests))
            , lifetimes(std::move(lifetimeTokens))
        {
        }

        NativeRequestBatch(NativeRequestBatch const&) = delete;
        NativeRequestBatch& operator=(NativeRequestBatch const&) = delete;

        NativeRequestBatch(NativeRequestBatch&& other) noexcept
            : requests(std::move(other.requests))
            , lifetimes(std::move(other.lifetimes))
            , m_ownsRequests(std::exchange(other.m_ownsRequests, false))
        {
        }

        NativeRequestBatch& operator=(NativeRequestBatch&&) = delete;

        ~NativeRequestBatch()
        {
            if(!m_ownsRequests)
                return;
            std::size_t offset = 0u;
            while(offset < requests.size())
            {
                auto const count
                    = static_cast<int>(std::min(requests.size() - offset, static_cast<std::size_t>(INT_MAX)));
                MPI_Waitall(count, requests.data() + offset, MPI_STATUSES_IGNORE);
                offset += static_cast<std::size_t>(count);
            }
        }

        std::vector<MPI_Request> requests;
        std::vector<std::shared_ptr<void>> lifetimes;

    private:
        void release() noexcept
        {
            m_ownsRequests = false;
        }

        bool m_ownsRequests = true;

        friend struct detail::NativeAccess;
    };

    namespace detail
    {
        struct NativeSubmission
        {
            std::function<NativeRequestBatch(NativeMpiContext&)> start;
            std::function<void(NativeMpiContext&, std::span<MPI_Status const>)> completed;
            std::function<void(std::exception_ptr)> failed;
            std::function<void()> cancelled;

            void setFailed(std::exception_ptr error) const
            {
                failed(std::move(error));
            }

            void cancel() const
            {
                cancelled();
            }
        };

        struct NativeBlockingSubmission
        {
            std::function<void(NativeMpiContext&)> invoke;
            std::function<void(std::exception_ptr)> failed;
            std::function<void()> cancelled;

            void setFailed(std::exception_ptr error) const
            {
                failed(std::move(error));
            }

            void cancel() const
            {
                cancelled();
            }
        };

        struct NativeAccess
        {
            static void release(NativeRequestBatch& batch)
            {
                batch.release();
            }

            static void submit(MpiExecutor& executor, Event predecessor, NativeSubmission submission);
            static void invokeBlocking(MpiExecutor& executor, Event predecessor, NativeBlockingSubmission submission);
        };

        struct NativeContextFactory;
    } // namespace detail

    /** MPI-native access valid only for the duration of an executor hook. */
    class NativeMpiContext
    {
    public:
        NativeMpiContext(NativeMpiContext const&) = delete;
        NativeMpiContext& operator=(NativeMpiContext const&) = delete;

        MPI_Comm communicator(CommunicatorId id) const
        {
            return m_resolve(m_implementation, id);
        }

        /** Transfer ownership of a newly created communicator to Caravan. */
        CommunicatorId adoptCommunicator(MPI_Comm communicator) const
        {
            return m_adopt(m_implementation, communicator);
        }

    private:
        using Resolve = MPI_Comm (*)(void*, CommunicatorId);
        using Adopt = CommunicatorId (*)(void*, MPI_Comm);

        NativeMpiContext(void* implementation, Resolve resolve, Adopt adopt)
            : m_implementation(implementation)
            , m_resolve(resolve)
            , m_adopt(adopt)
        {
        }

        void* m_implementation;
        Resolve m_resolve;
        Adopt m_adopt;

        friend struct detail::NativeContextFactory;
    };

    namespace detail
    {
        struct NativeContextFactory
        {
            static NativeMpiContext create(
                void* implementation,
                NativeMpiContext::Resolve resolve,
                NativeMpiContext::Adopt adopt)
            {
                return NativeMpiContext{implementation, resolve, adopt};
            }
        };
    } // namespace detail

    /** Submit native nonblocking MPI requests and return a typed result. */
    template<typename T, typename T_Start, typename T_Complete>
    Future<T> nativeFuture(MpiExecutor& executor, Event predecessor, T_Start&& start, T_Complete&& complete)
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);
        auto startWork = std::make_shared<std::decay_t<T_Start>>(std::forward<T_Start>(start));
        auto completeWork = std::make_shared<std::decay_t<T_Complete>>(std::forward<T_Complete>(complete));
        Promise<T> output;
        auto result = output.future();
        detail::NativeAccess::submit(
            executor,
            std::move(predecessor),
            detail::NativeSubmission{
                [startWork](NativeMpiContext& context) { return std::invoke(*startWork, context); },
                [completeWork, output](NativeMpiContext& context, std::span<MPI_Status const> statuses) mutable
                {
                    if constexpr(std::is_invocable_v<
                                     std::decay_t<T_Complete>&,
                                     NativeMpiContext&,
                                     std::span<MPI_Status const>>)
                        output.setValue(std::invoke(*completeWork, context, statuses));
                    else
                        output.setValue(std::invoke(*completeWork, statuses));
                },
                [output](std::exception_ptr error) mutable { output.setFailed(std::move(error)); },
                [output]() mutable { output.cancel(); }});
        return result;
    }

    /** Submit native nonblocking MPI requests without a result value. */
    template<typename T_Start, typename T_Complete>
    Event nativeEvent(MpiExecutor& executor, Event predecessor, T_Start&& start, T_Complete&& complete)
    {
        auto startWork = std::make_shared<std::decay_t<T_Start>>(std::forward<T_Start>(start));
        auto completeWork = std::make_shared<std::decay_t<T_Complete>>(std::forward<T_Complete>(complete));
        EventSource output;
        auto result = output.event();
        detail::NativeAccess::submit(
            executor,
            std::move(predecessor),
            detail::NativeSubmission{
                [startWork](NativeMpiContext& context) { return std::invoke(*startWork, context); },
                [completeWork, output](NativeMpiContext& context, std::span<MPI_Status const> statuses) mutable
                {
                    if constexpr(std::is_invocable_v<
                                     std::decay_t<T_Complete>&,
                                     NativeMpiContext&,
                                     std::span<MPI_Status const>>)
                        std::invoke(*completeWork, context, statuses);
                    else
                        std::invoke(*completeWork, statuses);
                    output.setReady();
                },
                [output](std::exception_ptr error) mutable { output.setFailed(std::move(error)); },
                [output]() mutable { output.cancel(); }});
        return result;
    }

    /** Run a blocking MPI call exclusively after active requests drain. */
    template<typename T, typename T_Operation>
    Future<T> nativeBlockingFuture(MpiExecutor& executor, Event predecessor, T_Operation&& operation)
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);
        auto work = std::make_shared<std::decay_t<T_Operation>>(std::forward<T_Operation>(operation));
        Promise<T> output;
        auto result = output.future();
        detail::NativeAccess::invokeBlocking(
            executor,
            std::move(predecessor),
            detail::NativeBlockingSubmission{
                [work, output](NativeMpiContext& context) mutable { output.setValue(std::invoke(*work, context)); },
                [output](std::exception_ptr error) mutable { output.setFailed(std::move(error)); },
                [output]() mutable { output.cancel(); }});
        return result;
    }

    /** Run a blocking MPI call exclusively without a result value. */
    template<typename T_Operation>
    Event nativeBlockingEvent(MpiExecutor& executor, Event predecessor, T_Operation&& operation)
    {
        auto work = std::make_shared<std::decay_t<T_Operation>>(std::forward<T_Operation>(operation));
        EventSource output;
        auto result = output.event();
        detail::NativeAccess::invokeBlocking(
            executor,
            std::move(predecessor),
            detail::NativeBlockingSubmission{
                [work, output](NativeMpiContext& context) mutable
                {
                    std::invoke(*work, context);
                    output.setReady();
                },
                [output](std::exception_ptr error) mutable { output.setFailed(std::move(error)); },
                [output]() mutable { output.cancel(); }});
        return result;
    }
} // namespace caravan
