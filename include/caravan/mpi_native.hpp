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
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/mpi.hpp>
#include <caravan/sender.hpp>
#include <mpi.h>

namespace caravan
{
    class NativeMpiContext;

    namespace detail
    {
        struct NativeAccess;
    } // namespace detail

    /** Native requests and lifetime tokens transferred to the MPI context.
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
            std::function<void()> stopped;
            std::optional<CommunicatorId> collective;

            void setFailed(std::exception_ptr error) const
            {
                failed(std::move(error));
            }

            void setStopped() const
            {
                stopped();
            }
        };

        struct NativeBlockingSubmission
        {
            std::function<void(NativeMpiContext&)> invoke;
            std::function<void(std::exception_ptr)> failed;
            std::function<void()> stopped;
            std::optional<CommunicatorId> collective;

            void setFailed(std::exception_ptr error) const
            {
                failed(std::move(error));
            }

            void setStopped() const
            {
                stopped();
            }
        };

        struct NativeAccess
        {
            static void release(NativeRequestBatch& batch)
            {
                batch.release();
            }

            static void submit(MpiContext& context, Event predecessor, NativeSubmission submission);
            static void invokeBlocking(MpiContext& context, Event predecessor, NativeBlockingSubmission submission);
        };

        struct NativeContextFactory;
    } // namespace detail

    /** MPI-native access valid only for the duration of an MPI-context hook. */
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

        /** Destroy a communicator previously adopted by Caravan. */
        void destroyCommunicator(CommunicatorId communicator) const
        {
            m_destroy(m_implementation, communicator);
        }

    private:
        using Resolve = MPI_Comm (*)(void*, CommunicatorId);
        using Adopt = CommunicatorId (*)(void*, MPI_Comm);
        using Destroy = void (*)(void*, CommunicatorId);

        NativeMpiContext(void* implementation, Resolve resolve, Adopt adopt, Destroy destroy)
            : m_implementation(implementation)
            , m_resolve(resolve)
            , m_adopt(adopt)
            , m_destroy(destroy)
        {
        }

        void* m_implementation;
        Resolve m_resolve;
        Adopt m_adopt;
        Destroy m_destroy;

        friend struct detail::NativeContextFactory;
    };

    namespace detail
    {
        struct NativeContextFactory
        {
            static NativeMpiContext create(
                void* implementation,
                NativeMpiContext::Resolve resolve,
                NativeMpiContext::Adopt adopt,
                NativeMpiContext::Destroy destroy)
            {
                return NativeMpiContext{implementation, resolve, adopt, destroy};
            }
        };
    } // namespace detail

    namespace detail
    {
        NativeRequestBatch startSend(
            NativeMpiContext& context,
            BufferLease const& buffer,
            Peer destination,
            MessageTag tag,
            CommunicatorId communicator);

        NativeRequestBatch startReceive(
            NativeMpiContext& context,
            BufferLease const& buffer,
            Peer source,
            MessageTag tag,
            CommunicatorId communicator);

        ReceiveResult completeReceive(std::span<MPI_Status const> statuses);

        NativeRequestBatch startAllReduce(
            NativeMpiContext& context,
            BufferLease const& input,
            BufferLease const& output,
            ScalarType type,
            ReduceOperation operation,
            CommunicatorId communicator,
            std::shared_ptr<std::size_t> const& elements);

        NativeRequestBatch startReduce(
            NativeMpiContext& context,
            BufferLease const& input,
            BufferLease const& output,
            ScalarType type,
            ReduceOperation operation,
            Peer root,
            CommunicatorId communicator,
            std::shared_ptr<std::size_t> const& elements);

        NativeRequestBatch startGather(
            NativeMpiContext& context,
            BufferLease const& input,
            BufferLease const& output,
            Peer root,
            CommunicatorId communicator,
            std::shared_ptr<std::size_t> const& resultBytes);

        NativeRequestBatch startGatherV(
            NativeMpiContext& context,
            BufferLease const& input,
            BufferLease const& output,
            std::vector<std::size_t> const& receiveBytes,
            std::vector<std::size_t> const& displacements,
            Peer root,
            CommunicatorId communicator,
            std::shared_ptr<std::size_t> const& resultBytes);

        NativeRequestBatch startBarrier(NativeMpiContext& context, CommunicatorId communicator);

        TopologySnapshot createCartesian(
            NativeMpiContext& context,
            std::vector<int> dimensions,
            std::vector<bool> periodic,
            int worldSize,
            int hostLocalRank);

        CommunicatorId duplicateCommunicator(NativeMpiContext& context, CommunicatorId communicator);

        std::optional<CommunicatorInfo> splitCommunicator(
            NativeMpiContext& context,
            std::optional<int> color,
            int key,
            CommunicatorId communicator);

        void destroyCommunicator(NativeMpiContext& context, CommunicatorId communicator);
    } // namespace detail

    namespace mpi
    {
        /** Lazy sender for one or more native nonblocking MPI requests. */
        template<typename T, typename T_Start, typename T_Complete>
        class RequestSender
        {
            static_assert(std::is_void_v<T> || (!std::is_reference_v<T> && !std::is_const_v<T>) );

        public:
            using completion_signatures
                = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<T>>;

            RequestSender(
                MpiContext& context,
                T_Start start,
                T_Complete complete,
                std::optional<CommunicatorId> collective)
                : m_context(&context)
                , m_start(std::move(start))
                , m_complete(std::move(complete))
                , m_collective(collective)
            {
            }

            template<typename T_Receiver>
            class Operation
            {
            public:
                Operation(
                    MpiContext& context,
                    T_Start start,
                    T_Complete complete,
                    T_Receiver receiver,
                    std::optional<CommunicatorId> collective)
                    : m_context(&context)
                    , m_start(std::move(start))
                    , m_complete(std::move(complete))
                    , m_receiver(std::move(receiver))
                    , m_collective(collective)
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    if(std::exchange(m_started, true))
                        std::terminate();

                    detail::NativeAccess::submit(
                        *m_context,
                        readyEvent(),
                        detail::NativeSubmission{
                            [this](NativeMpiContext& context) { return std::invoke(m_start, context); },
                            [this](NativeMpiContext& context, std::span<MPI_Status const> statuses)
                            {
                                if constexpr(std::is_void_v<T>)
                                {
                                    if constexpr(std::is_invocable_v<
                                                     T_Complete&,
                                                     NativeMpiContext&,
                                                     std::span<MPI_Status const>>)
                                        std::invoke(m_complete, context, statuses);
                                    else
                                        std::invoke(m_complete, statuses);
                                    m_receiver.set_value();
                                }
                                else if constexpr(std::is_invocable_v<
                                                      T_Complete&,
                                                      NativeMpiContext&,
                                                      std::span<MPI_Status const>>)
                                    m_receiver.set_value(std::invoke(m_complete, context, statuses));
                                else
                                    m_receiver.set_value(std::invoke(m_complete, statuses));
                            },
                            [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); },
                            [this] { m_receiver.set_stopped(); },
                            m_collective});
                }

            private:
                MpiContext* m_context;
                T_Start m_start;
                T_Complete m_complete;
                T_Receiver m_receiver;
                std::optional<CommunicatorId> m_collective;
                bool m_started = false;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{
                    *m_context,
                    std::move(m_start),
                    std::move(m_complete),
                    std::forward<T_Receiver>(receiver),
                    m_collective};
            }

        private:
            MpiContext* m_context;
            T_Start m_start;
            T_Complete m_complete;
            std::optional<CommunicatorId> m_collective;
        };

        /** Describe native MPI work without initiating it until operation start.
         *
         * Pass a communicator when the request initiates a collective; Caravan
         * then preserves managed collective initiation order on that communicator.
         */
        template<typename T, typename T_Start, typename T_Complete>
        auto request(
            MpiContext& context,
            T_Start&& start,
            T_Complete&& complete,
            std::optional<CommunicatorId> collective = {})
        {
            return RequestSender<T, std::decay_t<T_Start>, std::decay_t<T_Complete>>{
                context,
                std::forward<T_Start>(start),
                std::forward<T_Complete>(complete),
                collective};
        }

        inline auto send(
            MpiContext& context,
            BufferLease buffer,
            Peer destination,
            MessageTag tag,
            CommunicatorId communicator = worldCommunicator)
        {
            auto const bytes = buffer.bytes();
            return request<SendResult>(
                context,
                [buffer = std::move(buffer), destination, tag, communicator](NativeMpiContext& context)
                { return detail::startSend(context, buffer, destination, tag, communicator); },
                [bytes](std::span<MPI_Status const>) { return SendResult{bytes}; });
        }

        inline auto receive(
            MpiContext& context,
            BufferLease buffer,
            Peer source,
            MessageTag tag,
            CommunicatorId communicator = worldCommunicator)
        {
            return request<ReceiveResult>(
                context,
                [buffer = std::move(buffer), source, tag, communicator](NativeMpiContext& context)
                { return detail::startReceive(context, buffer, source, tag, communicator); },
                [](std::span<MPI_Status const> statuses) { return detail::completeReceive(statuses); });
        }

        inline auto allReduce(
            MpiContext& context,
            BufferLease input,
            BufferLease output,
            ScalarType type,
            ReduceOperation operation,
            CommunicatorId communicator = worldCommunicator)
        {
            auto elements = std::make_shared<std::size_t>(0u);
            return request<AllReduceResult>(
                context,
                [input = std::move(input), output = std::move(output), type, operation, communicator, elements](
                    NativeMpiContext& context)
                { return detail::startAllReduce(context, input, output, type, operation, communicator, elements); },
                [elements](std::span<MPI_Status const>) { return AllReduceResult{*elements}; },
                communicator);
        }

        inline auto reduce(
            MpiContext& context,
            BufferLease input,
            BufferLease output,
            ScalarType type,
            ReduceOperation operation,
            Peer root,
            CommunicatorId communicator = worldCommunicator)
        {
            auto elements = std::make_shared<std::size_t>(0u);
            return request<ReduceResult>(
                context,
                [input = std::move(input), output = std::move(output), type, operation, root, communicator, elements](
                    NativeMpiContext& context)
                { return detail::startReduce(context, input, output, type, operation, root, communicator, elements); },
                [elements](std::span<MPI_Status const>) { return ReduceResult{*elements}; },
                communicator);
        }

        inline auto gather(
            MpiContext& context,
            BufferLease input,
            BufferLease output,
            Peer root,
            CommunicatorId communicator = worldCommunicator)
        {
            auto resultBytes = std::make_shared<std::size_t>(0u);
            return request<GatherResult>(
                context,
                [input = std::move(input), output = std::move(output), root, communicator, resultBytes](
                    NativeMpiContext& context)
                { return detail::startGather(context, input, output, root, communicator, resultBytes); },
                [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
                communicator);
        }

        inline auto gatherV(
            MpiContext& context,
            BufferLease input,
            BufferLease output,
            std::vector<std::size_t> receiveBytes,
            std::vector<std::size_t> displacements,
            Peer root,
            CommunicatorId communicator = worldCommunicator)
        {
            auto resultBytes = std::make_shared<std::size_t>(0u);
            return request<GatherResult>(
                context,
                [input = std::move(input),
                 output = std::move(output),
                 receiveBytes = std::move(receiveBytes),
                 displacements = std::move(displacements),
                 root,
                 communicator,
                 resultBytes](NativeMpiContext& context)
                {
                    return detail::startGatherV(
                        context,
                        input,
                        output,
                        receiveBytes,
                        displacements,
                        root,
                        communicator,
                        resultBytes);
                },
                [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
                communicator);
        }

        inline auto barrier(MpiContext& context, CommunicatorId communicator = worldCommunicator)
        {
            return request<void>(
                context,
                [communicator](NativeMpiContext& context) { return detail::startBarrier(context, communicator); },
                [](std::span<MPI_Status const>) {},
                communicator);
        }

        template<typename T, typename T_Operation, bool T_Blocking>
        class ContextSender
        {
            static_assert(std::is_void_v<T> || (!std::is_reference_v<T> && !std::is_const_v<T>) );

        public:
            using completion_signatures
                = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<T>>;

            ContextSender(MpiContext& context, T_Operation operation, std::optional<CommunicatorId> collective = {})
                : m_context(&context)
                , m_operation(std::move(operation))
                , m_collective(collective)
            {
            }

            template<typename T_Receiver>
            class Operation
            {
            public:
                Operation(
                    MpiContext& context,
                    T_Operation operation,
                    T_Receiver receiver,
                    std::optional<CommunicatorId> collective)
                    : m_context(&context)
                    , m_operation(std::move(operation))
                    , m_receiver(std::move(receiver))
                    , m_collective(collective)
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    if(std::exchange(m_started, true))
                        std::terminate();

                    if constexpr(T_Blocking)
                        detail::NativeAccess::invokeBlocking(
                            *m_context,
                            readyEvent(),
                            detail::NativeBlockingSubmission{
                                [this](NativeMpiContext& context) { complete(context); },
                                [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); },
                                [this] { m_receiver.set_stopped(); },
                                m_collective});
                    else
                        detail::NativeAccess::submit(
                            *m_context,
                            readyEvent(),
                            detail::NativeSubmission{
                                [](NativeMpiContext&) { return NativeRequestBatch{}; },
                                [this](NativeMpiContext& context, std::span<MPI_Status const>) { complete(context); },
                                [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); },
                                [this] { m_receiver.set_stopped(); },
                                m_collective});
                }

            private:
                void complete(NativeMpiContext& context)
                {
                    if constexpr(std::is_void_v<T>)
                    {
                        std::invoke(m_operation, context);
                        m_receiver.set_value();
                    }
                    else
                        m_receiver.set_value(std::invoke(m_operation, context));
                }

                MpiContext* m_context;
                T_Operation m_operation;
                T_Receiver m_receiver;
                std::optional<CommunicatorId> m_collective;
                bool m_started = false;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{
                    *m_context,
                    std::move(m_operation),
                    std::forward<T_Receiver>(receiver),
                    m_collective};
            }

        private:
            MpiContext* m_context;
            T_Operation m_operation;
            std::optional<CommunicatorId> m_collective;
        };

        /** Lazily invoke a short operation on the MPI authority. */
        template<typename T_Operation>
        auto invoke(MpiContext& context, T_Operation&& operation, std::optional<CommunicatorId> collective = {})
        {
            using Operation = std::decay_t<T_Operation>;
            using Result
                = std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<Operation&, NativeMpiContext&>>>;
            return ContextSender<Result, Operation, false>{context, std::forward<T_Operation>(operation), collective};
        }

        /** Lazily invoke a blocking operation without draining unrelated requests. */
        template<typename T_Operation>
        auto invokeBlocking(
            MpiContext& context,
            T_Operation&& operation,
            std::optional<CommunicatorId> collective = {})
        {
            using Operation = std::decay_t<T_Operation>;
            using Result
                = std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<Operation&, NativeMpiContext&>>>;
            return ContextSender<Result, Operation, true>{context, std::forward<T_Operation>(operation), collective};
        }

        inline auto createCartesian(MpiContext& context, std::vector<int> dimensions, std::vector<bool> periodic)
        {
            auto const topology = context.topology();
            return invokeBlocking(
                context,
                [dimensions = std::move(dimensions),
                 periodic = std::move(periodic),
                 worldSize = topology.size,
                 hostLocalRank = topology.hostLocalRank](NativeMpiContext& context) mutable
                {
                    return detail::createCartesian(
                        context,
                        std::move(dimensions),
                        std::move(periodic),
                        worldSize,
                        hostLocalRank);
                },
                worldCommunicator);
        }

        inline auto duplicateCommunicator(MpiContext& context, CommunicatorId communicator = worldCommunicator)
        {
            return invokeBlocking(
                context,
                [communicator](NativeMpiContext& context)
                { return detail::duplicateCommunicator(context, communicator); },
                communicator);
        }

        inline auto splitCommunicator(
            MpiContext& context,
            std::optional<int> color,
            int key,
            CommunicatorId communicator = worldCommunicator)
        {
            return invokeBlocking(
                context,
                [color, key, communicator](NativeMpiContext& context)
                { return detail::splitCommunicator(context, color, key, communicator); },
                communicator);
        }

        inline auto destroyCommunicator(MpiContext& context, CommunicatorId communicator)
        {
            return invokeBlocking(
                context,
                [communicator](NativeMpiContext& context) { detail::destroyCommunicator(context, communicator); },
                communicator);
        }
    } // namespace mpi

    /** Submit native nonblocking MPI requests and return a typed result.
     *
     * The optional communicator marks managed collective initiation ordering.
     */
    template<typename T, typename T_Start, typename T_Complete>
    Future<T> nativeFuture(
        MpiContext& context,
        Event predecessor,
        T_Start&& start,
        T_Complete&& complete,
        std::optional<CommunicatorId> collective = {})
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);
        auto startWork = std::make_shared<std::decay_t<T_Start>>(std::forward<T_Start>(start));
        auto completeWork = std::make_shared<std::decay_t<T_Complete>>(std::forward<T_Complete>(complete));
        Promise<T> output;
        auto result = output.future();
        detail::NativeAccess::submit(
            context,
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
                [output]() mutable { output.setStopped(); },
                collective});
        return result;
    }

    /** Submit native nonblocking MPI requests without a result value. */
    template<typename T_Start, typename T_Complete>
    Event nativeEvent(
        MpiContext& context,
        Event predecessor,
        T_Start&& start,
        T_Complete&& complete,
        std::optional<CommunicatorId> collective = {})
    {
        auto startWork = std::make_shared<std::decay_t<T_Start>>(std::forward<T_Start>(start));
        auto completeWork = std::make_shared<std::decay_t<T_Complete>>(std::forward<T_Complete>(complete));
        EventSource output;
        auto result = output.event();
        detail::NativeAccess::submit(
            context,
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
                [output]() mutable { output.setStopped(); },
                collective});
        return result;
    }

    /** Run a blocking MPI call without implicitly draining unrelated requests.
     *
     * The optional communicator marks managed collective initiation ordering.
     */
    template<typename T, typename T_Operation>
    Future<T> nativeBlockingFuture(
        MpiContext& context,
        Event predecessor,
        T_Operation&& operation,
        std::optional<CommunicatorId> collective = {})
    {
        static_assert(!std::is_void_v<T> && !std::is_reference_v<T>);
        auto work = std::make_shared<std::decay_t<T_Operation>>(std::forward<T_Operation>(operation));
        Promise<T> output;
        auto result = output.future();
        detail::NativeAccess::invokeBlocking(
            context,
            std::move(predecessor),
            detail::NativeBlockingSubmission{
                [work, output](NativeMpiContext& context) mutable { output.setValue(std::invoke(*work, context)); },
                [output](std::exception_ptr error) mutable { output.setFailed(std::move(error)); },
                [output]() mutable { output.setStopped(); },
                collective});
        return result;
    }

    /** Run a blocking MPI call without a result or implicit global quiescence. */
    template<typename T_Operation>
    Event nativeBlockingEvent(
        MpiContext& context,
        Event predecessor,
        T_Operation&& operation,
        std::optional<CommunicatorId> collective = {})
    {
        auto work = std::make_shared<std::decay_t<T_Operation>>(std::forward<T_Operation>(operation));
        EventSource output;
        auto result = output.event();
        detail::NativeAccess::invokeBlocking(
            context,
            std::move(predecessor),
            detail::NativeBlockingSubmission{
                [work, output](NativeMpiContext& context) mutable
                {
                    std::invoke(*work, context);
                    output.setReady();
                },
                [output](std::exception_ptr error) mutable { output.setFailed(std::move(error)); },
                [output]() mutable { output.setStopped(); },
                collective});
        return result;
    }
} // namespace caravan
