/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <exception>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <caravan/core/sender.hpp>
#include <caravan/mpi/context.hpp>

namespace caravan::mpi
{
    namespace operation_detail
    {
        template<typename T>
        struct ValueCallback
        {
            using type = std::function<void(T)>;
        };

        template<>
        struct ValueCallback<void>
        {
            using type = std::function<void()>;
        };

        using ErrorCallback = std::function<void(std::exception_ptr)>;
        using StoppedCallback = std::function<void()>;

        struct Send
        {
            ConstBufferLease buffer;
            Peer peer;
            MessageTag tag;
            CommunicatorId communicator;
        };

        struct Receive
        {
            BufferLease buffer;
            Peer peer;
            MessageTag tag;
            CommunicatorId communicator;
        };

        struct AllReduce
        {
            ConstBufferLease input;
            BufferLease output;
            ScalarType type;
            ReduceOperation operation;
            CommunicatorId communicator;
        };

        struct Reduce
        {
            ConstBufferLease input;
            BufferLease output;
            ScalarType type;
            ReduceOperation operation;
            Peer root;
            CommunicatorId communicator;
        };

        struct Gather
        {
            ConstBufferLease input;
            BufferLease output;
            Peer root;
            CommunicatorId communicator;
        };

        struct AllGather
        {
            ConstBufferLease input;
            BufferLease output;
            CommunicatorId communicator;
        };

        struct GatherV
        {
            ConstBufferLease input;
            BufferLease output;
            std::vector<std::size_t> receiveBytes;
            std::vector<std::size_t> displacements;
            Peer root;
            CommunicatorId communicator;
        };

        struct Barrier
        {
            CommunicatorId communicator;
        };

        struct CreateCartesian
        {
            std::vector<int> dimensions;
            std::vector<bool> periodic;
            int worldSize;
            int hostLocalRank;
        };

        struct DuplicateCommunicator
        {
            CommunicatorId communicator;
        };

        struct SplitCommunicator
        {
            std::optional<int> color;
            int key;
            CommunicatorId communicator;
        };

        struct DestroyCommunicator
        {
            CommunicatorId communicator;
        };

        template<typename T>
        struct Descriptor;

        template<>
        struct Descriptor<SendResult>
        {
            using type = std::variant<Send>;
        };

        template<>
        struct Descriptor<ReceiveResult>
        {
            using type = std::variant<Receive>;
        };

        template<>
        struct Descriptor<AllReduceResult>
        {
            using type = std::variant<AllReduce>;
        };

        template<>
        struct Descriptor<ReduceResult>
        {
            using type = std::variant<Reduce>;
        };

        template<>
        struct Descriptor<GatherResult>
        {
            using type = std::variant<Gather, AllGather, GatherV>;
        };

        template<>
        struct Descriptor<void>
        {
            using type = std::variant<Barrier, DestroyCommunicator>;
        };

        template<>
        struct Descriptor<TopologySnapshot>
        {
            using type = std::variant<CreateCartesian>;
        };

        template<>
        struct Descriptor<CommunicatorId>
        {
            using type = std::variant<DuplicateCommunicator>;
        };

        template<>
        struct Descriptor<std::optional<CommunicatorInfo>>
        {
            using type = std::variant<SplitCommunicator>;
        };

#define CARAVAN_DECLARE_MPI_SUBMIT(Result, Operation)                                                                 \
    void submit(MpiContext&, Operation, ValueCallback<Result>::type, ErrorCallback, StoppedCallback)

        CARAVAN_DECLARE_MPI_SUBMIT(SendResult, Send);
        CARAVAN_DECLARE_MPI_SUBMIT(ReceiveResult, Receive);
        CARAVAN_DECLARE_MPI_SUBMIT(AllReduceResult, AllReduce);
        CARAVAN_DECLARE_MPI_SUBMIT(ReduceResult, Reduce);
        CARAVAN_DECLARE_MPI_SUBMIT(GatherResult, Gather);
        CARAVAN_DECLARE_MPI_SUBMIT(GatherResult, AllGather);
        CARAVAN_DECLARE_MPI_SUBMIT(GatherResult, GatherV);
        CARAVAN_DECLARE_MPI_SUBMIT(void, Barrier);
        CARAVAN_DECLARE_MPI_SUBMIT(TopologySnapshot, CreateCartesian);
        CARAVAN_DECLARE_MPI_SUBMIT(CommunicatorId, DuplicateCommunicator);
        CARAVAN_DECLARE_MPI_SUBMIT(std::optional<CommunicatorInfo>, SplitCommunicator);
        CARAVAN_DECLARE_MPI_SUBMIT(void, DestroyCommunicator);

#undef CARAVAN_DECLARE_MPI_SUBMIT
    } // namespace operation_detail

    /** Allocation-free description of one ordinary MPI operation.
     *
     * Native MPI details and queue callback erasure remain behind the MPI backend
     * boundary; constructing and connecting this sender only moves concrete state.
     */
    template<typename T>
    class OperationSender
    {
        static_assert(std::is_void_v<T> || (!std::is_reference_v<T> && !std::is_const_v<T>) );

        using Descriptor = typename operation_detail::Descriptor<T>::type;
        using ValueCallback = typename operation_detail::ValueCallback<T>::type;

    public:
        using completion_signatures
            = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<T>>;

        template<typename T_Descriptor>
        OperationSender(MpiContext& context, T_Descriptor descriptor)
            : m_context(&context)
            , m_descriptor(std::move(descriptor))
        {
        }

        template<typename T_Receiver>
        class Operation
        {
        public:
            Operation(MpiContext& context, Descriptor descriptor, T_Receiver receiver)
                : m_context(&context)
                , m_descriptor(std::move(descriptor))
                , m_receiver(std::move(receiver))
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

                try
                {
                    ValueCallback value;
                    if constexpr(std::is_void_v<T>)
                        value = [this] { m_receiver.set_value(); };
                    else
                        value = [this](T result) { m_receiver.set_value(std::move(result)); };
                    std::visit(
                        [this, value = std::move(value)](auto descriptor) mutable
                        {
                            operation_detail::submit(
                                *m_context,
                                std::move(descriptor),
                                std::move(value),
                                [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); },
                                [this] { m_receiver.set_stopped(); });
                        },
                        std::move(m_descriptor));
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

        private:
            MpiContext* m_context;
            Descriptor m_descriptor;
            T_Receiver m_receiver;
            bool m_started = false;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{
                *m_context,
                std::move(m_descriptor),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        MpiContext* m_context;
        Descriptor m_descriptor;
    };

    OperationSender<SendResult> send(
        MpiContext& context,
        ConstBufferLease buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<ReceiveResult> receive(
        MpiContext& context,
        BufferLease buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<AllReduceResult> allReduce(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<ReduceResult> reduce(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<GatherResult> gather(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<GatherResult> allGather(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<GatherResult> gatherV(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        std::vector<std::size_t> receiveBytes,
        std::vector<std::size_t> displacements,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<void> barrier(MpiContext& context, CommunicatorId communicator = worldCommunicator);

    OperationSender<TopologySnapshot> createCartesian(
        MpiContext& context,
        std::vector<int> dimensions,
        std::vector<bool> periodic);

    OperationSender<CommunicatorId> duplicateCommunicator(
        MpiContext& context,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<std::optional<CommunicatorInfo>> splitCommunicator(
        MpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<void> destroyCommunicator(MpiContext& context, CommunicatorId communicator);
} // namespace caravan::mpi
