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
    } // namespace operation_detail

    template<typename T>
    class OperationSender
    {
        static_assert(std::is_void_v<T> || (!std::is_reference_v<T> && !std::is_const_v<T>) );

        using ValueCallback = typename operation_detail::ValueCallback<T>::type;
        using ErrorCallback = std::function<void(std::exception_ptr)>;
        using StoppedCallback = std::function<void()>;

    public:
        using completion_signatures
            = caravan::detail::DefaultCompletionSignatures<caravan::detail::ResultValueSignature<T>>;
        using Start = std::function<void(ValueCallback, ErrorCallback, StoppedCallback)>;

        explicit OperationSender(Start start) : m_start(std::move(start))
        {
        }

        template<typename T_Receiver>
        class Operation
        {
        public:
            Operation(Start start, T_Receiver receiver) : m_start(std::move(start)), m_receiver(std::move(receiver))
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
                    auto start = std::move(m_start);
                    if constexpr(std::is_void_v<T>)
                        start(
                            [this] { m_receiver.set_value(); },
                            [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); },
                            [this] { m_receiver.set_stopped(); });
                    else
                        start(
                            [this](T value) { m_receiver.set_value(std::move(value)); },
                            [this](std::exception_ptr error) { m_receiver.set_error(std::move(error)); },
                            [this] { m_receiver.set_stopped(); });
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

        private:
            Start m_start;
            T_Receiver m_receiver;
            bool m_started = false;
        };

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return Operation<std::decay_t<T_Receiver>>{std::move(m_start), std::forward<T_Receiver>(receiver)};
        }

    private:
        Start m_start;
    };

    OperationSender<SendResult> send(
        MpiContext& context,
        BufferLease buffer,
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
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<ReduceResult> reduce(
        MpiContext& context,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<GatherResult> gather(
        MpiContext& context,
        BufferLease input,
        BufferLease output,
        Peer root,
        CommunicatorId communicator = worldCommunicator);

    OperationSender<GatherResult> gatherV(
        MpiContext& context,
        BufferLease input,
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
