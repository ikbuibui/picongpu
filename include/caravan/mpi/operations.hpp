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

        struct Send
        {
            using Result = SendResult;

            ConstBufferLease buffer;
            Peer peer;
            MessageTag tag;
            CommunicatorId communicator;
        };

        struct Receive
        {
            using Result = ReceiveResult;

            BufferLease buffer;
            Peer peer;
            MessageTag tag;
            CommunicatorId communicator;
        };

        struct AllReduce
        {
            using Result = AllReduceResult;

            ConstBufferLease input;
            BufferLease output;
            ScalarType type;
            ReduceOperation operation;
            CommunicatorId communicator;
        };

        struct Reduce
        {
            using Result = ReduceResult;

            ConstBufferLease input;
            BufferLease output;
            ScalarType type;
            ReduceOperation operation;
            Peer root;
            CommunicatorId communicator;
        };

        struct Gather
        {
            using Result = GatherResult;

            ConstBufferLease input;
            BufferLease output;
            Peer root;
            CommunicatorId communicator;
        };

        struct AllGather
        {
            using Result = GatherResult;

            ConstBufferLease input;
            BufferLease output;
            CommunicatorId communicator;
        };

        struct GatherV
        {
            using Result = GatherResult;

            ConstBufferLease input;
            BufferLease output;
            std::vector<std::size_t> receiveBytes;
            std::vector<std::size_t> displacements;
            Peer root;
            CommunicatorId communicator;
        };

        struct Barrier
        {
            using Result = void;

            CommunicatorId communicator;
        };

        struct CreateCartesian
        {
            using Result = TopologySnapshot;

            std::vector<int> dimensions;
            std::vector<bool> periodic;
            int worldSize;
            int hostLocalRank;
        };

        struct DuplicateCommunicator
        {
            using Result = CommunicatorId;

            CommunicatorId communicator;
        };

        struct SplitCommunicator
        {
            using Result = std::optional<CommunicatorInfo>;

            std::optional<int> color;
            int key;
            CommunicatorId communicator;
        };

        struct DestroyCommunicator
        {
            using Result = void;

            CommunicatorId communicator;
        };

#define CARAVAN_DECLARE_MPI_SUBMIT(Operation)                                                                         \
    void submit(MpiContext&, Operation, ValueCallback<Operation::Result>::type, ErrorCallback)

        CARAVAN_DECLARE_MPI_SUBMIT(Send);
        CARAVAN_DECLARE_MPI_SUBMIT(Receive);
        CARAVAN_DECLARE_MPI_SUBMIT(AllReduce);
        CARAVAN_DECLARE_MPI_SUBMIT(Reduce);
        CARAVAN_DECLARE_MPI_SUBMIT(Gather);
        CARAVAN_DECLARE_MPI_SUBMIT(AllGather);
        CARAVAN_DECLARE_MPI_SUBMIT(GatherV);
        CARAVAN_DECLARE_MPI_SUBMIT(Barrier);
        CARAVAN_DECLARE_MPI_SUBMIT(CreateCartesian);
        CARAVAN_DECLARE_MPI_SUBMIT(DuplicateCommunicator);
        CARAVAN_DECLARE_MPI_SUBMIT(SplitCommunicator);
        CARAVAN_DECLARE_MPI_SUBMIT(DestroyCommunicator);

#undef CARAVAN_DECLARE_MPI_SUBMIT
    } // namespace operation_detail

    /** Allocation-free description of one ordinary MPI operation.
     *
     * Native MPI details and queue callback erasure remain behind the MPI backend
     * boundary; constructing and connecting this sender only moves concrete state.
     */
    template<typename T_Descriptor>
    class OperationSender
    {
        using Result = typename T_Descriptor::Result;
        using ValueCallback = typename operation_detail::ValueCallback<Result>::type;

        static_assert(std::is_void_v<Result> || (!std::is_reference_v<Result> && !std::is_const_v<Result>) );

    public:
        using completion_signatures
            = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<Result>>;

        OperationSender(MpiContext& context, T_Descriptor descriptor)
            : m_context(&context)
            , m_descriptor(std::move(descriptor))
        {
        }

        template<typename T_Receiver>
        class Operation
        {
        public:
            Operation(MpiContext& context, T_Descriptor descriptor, T_Receiver receiver)
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
                    if constexpr(std::is_void_v<Result>)
                        value = [this] { m_receiver.set_value(); };
                    else
                        value = [this](Result result) { m_receiver.set_value(std::move(result)); };
                    operation_detail::submit(
                        *m_context,
                        std::move(m_descriptor),
                        std::move(value),
                        [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

        private:
            MpiContext* m_context;
            T_Descriptor m_descriptor;
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

        /** Whether start() immediately submits a collective for this lane. */
        bool managedCollectiveOn(MpiContext const& context, CommunicatorId communicator) const noexcept
        {
            if(m_context != &context)
                return false;
            if constexpr(std::is_same_v<T_Descriptor, operation_detail::CreateCartesian>)
                return communicator == worldCommunicator;
            else if constexpr(
                std::is_same_v<T_Descriptor, operation_detail::Send>
                || std::is_same_v<T_Descriptor, operation_detail::Receive>)
                return false;
            else
                return m_descriptor.communicator == communicator;
        }

    private:
        MpiContext* m_context;
        T_Descriptor m_descriptor;
    };

    OperationSender<operation_detail::Send> send(
        MpiContext& context,
        ConstBufferLease buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::Receive> receive(
        MpiContext& context,
        BufferLease buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::AllReduce> allReduce(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::Reduce> reduce(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::Gather> gather(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::AllGather> allGather(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::GatherV> gatherV(
        MpiContext& context,
        ConstBufferLease input,
        BufferLease output,
        std::vector<std::size_t> receiveBytes,
        std::vector<std::size_t> displacements,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::Barrier> barrier(
        MpiContext& context,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::CreateCartesian> createCartesian(
        MpiContext& context,
        std::vector<int> dimensions,
        std::vector<bool> periodic);

    OperationSender<operation_detail::DuplicateCommunicator> duplicateCommunicator(
        MpiContext& context,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::SplitCommunicator> splitCommunicator(
        MpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<operation_detail::DestroyCommunicator> destroyCommunicator(
        MpiContext& context,
        CommunicatorId communicator);
} // namespace caravan::mpi
