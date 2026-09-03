/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <climits>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/mpi/error.hpp>
#include <caravan/mpi/native.hpp>
#include <caravan/mpi/operations.hpp>
#include <mpi.h>

namespace caravan
{
    using detail::mpiError;

    namespace
    {
        bool validBuffer(BufferLease const& buffer)
        {
            return buffer.valid() && buffer.bytes() <= static_cast<std::size_t>(INT_MAX);
        }

        std::size_t scalarSize(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::int32:
            case ScalarType::uint32:
            case ScalarType::float32:
                return 4u;
            case ScalarType::int64:
            case ScalarType::uint64:
            case ScalarType::float64:
                return 8u;
            }
            return 0u;
        }

        MPI_Datatype nativeType(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::int32:
                return MPI_INT32_T;
            case ScalarType::uint32:
                return MPI_UINT32_T;
            case ScalarType::int64:
                return MPI_INT64_T;
            case ScalarType::uint64:
                return MPI_UINT64_T;
            case ScalarType::float32:
                return MPI_FLOAT;
            case ScalarType::float64:
                return MPI_DOUBLE;
            }
            throw std::invalid_argument("Unknown Caravan scalar type");
        }

        MPI_Op nativeOperation(ReduceOperation operation)
        {
            switch(operation)
            {
            case ReduceOperation::sum:
                return MPI_SUM;
            case ReduceOperation::minimum:
                return MPI_MIN;
            case ReduceOperation::maximum:
                return MPI_MAX;
            case ReduceOperation::product:
                return MPI_PROD;
            }
            throw std::invalid_argument("Unknown Caravan reduce operation");
        }

        template<typename T, typename T_Start, typename T_Complete>
        void submitRequest(
            MpiContext& context,
            T_Start start,
            T_Complete complete,
            typename mpi::operation_detail::ValueCallback<T>::type value,
            mpi::operation_detail::ErrorCallback error,
            mpi::operation_detail::StoppedCallback stopped,
            std::optional<CommunicatorId> collective = {})
        {
            detail::NativeAccess::submit(
                context,
                detail::NativeSubmission{
                    [start = std::move(start)](NativeMpiContext& native) mutable
                    { return detail::invokeNative(start, native); },
                    [complete = std::move(complete),
                     value = std::move(value)](NativeMpiContext& native, std::span<MPI_Status const> statuses) mutable
                    {
                        if constexpr(std::is_void_v<T>)
                        {
                            if constexpr(std::is_invocable_v<
                                             T_Complete&,
                                             NativeMpiContext&,
                                             std::span<MPI_Status const>>)
                                detail::invokeNative(complete, native, statuses);
                            else
                                detail::invokeNative(complete, statuses);
                            value();
                        }
                        else if constexpr(std::is_invocable_v<
                                              T_Complete&,
                                              NativeMpiContext&,
                                              std::span<MPI_Status const>>)
                            value(detail::invokeNative(complete, native, statuses));
                        else
                            value(detail::invokeNative(complete, statuses));
                    },
                    std::move(error),
                    std::move(stopped),
                    collective});
        }

        template<typename T, typename T_Operation>
        void submitBlocking(
            MpiContext& context,
            T_Operation operation,
            typename mpi::operation_detail::ValueCallback<T>::type value,
            mpi::operation_detail::ErrorCallback error,
            mpi::operation_detail::StoppedCallback stopped,
            std::optional<CommunicatorId> collective = {})
        {
            detail::NativeAccess::invokeBlocking(
                context,
                detail::NativeBlockingSubmission{
                    [operation = std::move(operation), value = std::move(value)](NativeMpiContext& native) mutable
                    {
                        if constexpr(std::is_void_v<T>)
                        {
                            detail::invokeNative(operation, native);
                            value();
                        }
                        else
                            value(detail::invokeNative(operation, native));
                    },
                    std::move(error),
                    std::move(stopped),
                    collective});
        }
    } // namespace

    NativeRequestBatch detail::startSend(
        NativeMpiContext& context,
        BufferLease const& buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || destination.any || destination.value < 0 || tag.any || tag.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI send");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.lifetime()});
        int const error = MPI_Isend(
            buffer.data(),
            static_cast<int>(buffer.bytes()),
            MPI_BYTE,
            destination.value,
            tag.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Isend", error);
        return batch;
    }

    NativeRequestBatch detail::startReceive(
        NativeMpiContext& context,
        BufferLease const& buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || (!source.any && source.value < 0) || (!tag.any && tag.value < 0))
            throw std::invalid_argument("Invalid Caravan MPI receive");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.lifetime()});
        int const error = MPI_Irecv(
            buffer.data(),
            static_cast<int>(buffer.bytes()),
            MPI_BYTE,
            source.any ? MPI_ANY_SOURCE : source.value,
            tag.any ? MPI_ANY_TAG : tag.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Irecv", error);
        return batch;
    }

    ReceiveResult detail::completeReceive(std::span<MPI_Status const> statuses)
    {
        int bytes = 0;
        int const error = MPI_Get_count(&statuses.front(), MPI_BYTE, &bytes);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Get_count", error);
        if(bytes == MPI_UNDEFINED)
            throw std::runtime_error("MPI_Get_count returned MPI_UNDEFINED");
        return ReceiveResult{
            Peer{statuses.front().MPI_SOURCE},
            MessageTag{statuses.front().MPI_TAG},
            static_cast<std::size_t>(bytes)};
    }

    NativeRequestBatch detail::startAllReduce(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& elements)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output) || input.bytes() % elementBytes != 0u
           || output.bytes() < input.bytes())
            throw std::invalid_argument("Invalid Caravan MPI all-reduce");
        *elements = input.bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.lifetime(), output.lifetime(), elements});
        void* const sendBuffer = input.data() == output.data() ? MPI_IN_PLACE : input.data();
        int const error = MPI_Iallreduce(
            sendBuffer,
            output.data(),
            static_cast<int>(*elements),
            nativeType(type),
            nativeOperation(operation),
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Iallreduce", error);
        return batch;
    }

    NativeRequestBatch detail::startReduce(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& elements)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output) || input.bytes() % elementBytes != 0u
           || output.bytes() < input.bytes() || input.data() == output.data() || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI reduce");
        *elements = input.bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.lifetime(), output.lifetime(), elements});
        int const error = MPI_Ireduce(
            input.data(),
            output.data(),
            static_cast<int>(*elements),
            nativeType(type),
            nativeOperation(operation),
            root.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ireduce", error);
        return batch;
    }

    NativeRequestBatch detail::startGather(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI gather");

        auto const native = context.communicator(communicator);
        int rank = -1;
        int size = 0;
        int error = MPI_Comm_rank(native, &rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI gather communicator query", error);
        if(rank == root.value)
        {
            if(size <= 0 || input.bytes() > output.bytes() / static_cast<std::size_t>(size))
                throw std::invalid_argument("Caravan MPI gather output is too small");
            *resultBytes = input.bytes() * static_cast<std::size_t>(size);
        }

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.lifetime(), output.lifetime(), resultBytes});
        error = MPI_Igather(
            input.data(),
            static_cast<int>(input.bytes()),
            MPI_BYTE,
            output.data(),
            static_cast<int>(input.bytes()),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igather", error);
        return batch;
    }

    NativeRequestBatch detail::startGatherV(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        std::vector<std::size_t> const& receiveBytes,
        std::vector<std::size_t> const& displacements,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || receiveBytes.size() != displacements.size() || root.any
           || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI variable gather");

        auto counts = std::make_shared<std::vector<int>>();
        auto offsets = std::make_shared<std::vector<int>>();
        counts->reserve(receiveBytes.size());
        offsets->reserve(displacements.size());
        for(std::size_t i = 0u; i < receiveBytes.size(); ++i)
        {
            if(receiveBytes[i] > static_cast<std::size_t>(INT_MAX)
               || displacements[i] > static_cast<std::size_t>(INT_MAX))
                throw std::invalid_argument("Invalid Caravan MPI variable gather layout");
            counts->emplace_back(static_cast<int>(receiveBytes[i]));
            offsets->emplace_back(static_cast<int>(displacements[i]));
        }

        auto const native = context.communicator(communicator);
        int rank = -1;
        int size = 0;
        int error = MPI_Comm_rank(native, &rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI variable gather communicator query", error);
        if(rank == root.value)
        {
            if(size <= 0 || counts->size() != static_cast<std::size_t>(size))
                throw std::invalid_argument("Invalid Caravan MPI variable gather rank count");
            for(std::size_t i = 0u; i < counts->size(); ++i)
            {
                auto const end = static_cast<std::size_t>((*offsets)[i]) + static_cast<std::size_t>((*counts)[i]);
                if(end > output.bytes())
                    throw std::invalid_argument("Caravan MPI variable gather output is too small");
                *resultBytes += static_cast<std::size_t>((*counts)[i]);
            }
        }

        NativeRequestBatch batch(
            {MPI_REQUEST_NULL},
            {input.lifetime(), output.lifetime(), counts, offsets, resultBytes});
        error = MPI_Igatherv(
            input.data(),
            static_cast<int>(input.bytes()),
            MPI_BYTE,
            output.data(),
            counts->data(),
            offsets->data(),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igatherv", error);
        return batch;
    }

    NativeRequestBatch detail::startBarrier(NativeMpiContext& context, CommunicatorId communicator)
    {
        NativeRequestBatch batch({MPI_REQUEST_NULL});
        int const error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ibarrier", error);
        return batch;
    }

    mpi::OperationSender<SendResult> mpi::send(
        MpiContext& context,
        BufferLease buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        return {context, operation_detail::Send{std::move(buffer), destination, tag, communicator}};
    }

    mpi::OperationSender<ReceiveResult> mpi::receive(
        MpiContext& context,
        BufferLease buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        return {context, operation_detail::Receive{std::move(buffer), source, tag, communicator}};
    }

    mpi::OperationSender<AllReduceResult> mpi::allReduce(
        MpiContext& context,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator)
    {
        return {
            context,
            operation_detail::AllReduce{std::move(input), std::move(output), type, operation, communicator}};
    }

    mpi::OperationSender<ReduceResult> mpi::reduce(
        MpiContext& context,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator)
    {
        return {
            context,
            operation_detail::Reduce{std::move(input), std::move(output), type, operation, root, communicator}};
    }

    mpi::OperationSender<GatherResult> mpi::gather(
        MpiContext& context,
        BufferLease input,
        BufferLease output,
        Peer root,
        CommunicatorId communicator)
    {
        return {context, operation_detail::Gather{std::move(input), std::move(output), root, communicator}};
    }

    mpi::OperationSender<GatherResult> mpi::gatherV(
        MpiContext& context,
        BufferLease input,
        BufferLease output,
        std::vector<std::size_t> receiveBytes,
        std::vector<std::size_t> displacements,
        Peer root,
        CommunicatorId communicator)
    {
        return {
            context,
            operation_detail::GatherV{
                std::move(input),
                std::move(output),
                std::move(receiveBytes),
                std::move(displacements),
                root,
                communicator}};
    }

    mpi::OperationSender<void> mpi::barrier(MpiContext& context, CommunicatorId communicator)
    {
        return {context, operation_detail::Barrier{communicator}};
    }

    mpi::OperationSender<TopologySnapshot> mpi::createCartesian(
        MpiContext& context,
        std::vector<int> dimensions,
        std::vector<bool> periodic)
    {
        auto const topology = context.topology();
        return {
            context,
            operation_detail::CreateCartesian{
                std::move(dimensions),
                std::move(periodic),
                topology.size,
                topology.hostLocalRank}};
    }

    mpi::OperationSender<CommunicatorId> mpi::duplicateCommunicator(MpiContext& context, CommunicatorId communicator)
    {
        return {context, operation_detail::DuplicateCommunicator{communicator}};
    }

    mpi::OperationSender<std::optional<CommunicatorInfo>> mpi::splitCommunicator(
        MpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator)
    {
        return {context, operation_detail::SplitCommunicator{color, key, communicator}};
    }

    mpi::OperationSender<void> mpi::destroyCommunicator(MpiContext& context, CommunicatorId communicator)
    {
        return {context, operation_detail::DestroyCommunicator{communicator}};
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Send operation,
        ValueCallback<SendResult>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        auto const bytes = operation.buffer.bytes();
        submitRequest<SendResult>(
            context,
            [operation = std::move(operation)](NativeMpiContext& native)
            {
                return detail::startSend(
                    native,
                    operation.buffer,
                    operation.peer,
                    operation.tag,
                    operation.communicator);
            },
            [bytes](std::span<MPI_Status const>) { return SendResult{bytes}; },
            std::move(value),
            std::move(error),
            std::move(stopped));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Receive operation,
        ValueCallback<ReceiveResult>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        submitRequest<ReceiveResult>(
            context,
            [operation = std::move(operation)](NativeMpiContext& native)
            {
                return detail::startReceive(
                    native,
                    operation.buffer,
                    operation.peer,
                    operation.tag,
                    operation.communicator);
            },
            [](std::span<MPI_Status const> statuses) { return detail::completeReceive(statuses); },
            std::move(value),
            std::move(error),
            std::move(stopped));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        AllReduce operation,
        ValueCallback<AllReduceResult>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        auto elements = std::make_shared<std::size_t>(0u);
        auto const communicator = operation.communicator;
        submitRequest<AllReduceResult>(
            context,
            [operation = std::move(operation), elements](NativeMpiContext& native)
            {
                return detail::startAllReduce(
                    native,
                    operation.input,
                    operation.output,
                    operation.type,
                    operation.operation,
                    operation.communicator,
                    elements);
            },
            [elements](std::span<MPI_Status const>) { return AllReduceResult{*elements}; },
            std::move(value),
            std::move(error),
            std::move(stopped),
            communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Reduce operation,
        ValueCallback<ReduceResult>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        auto elements = std::make_shared<std::size_t>(0u);
        auto const communicator = operation.communicator;
        submitRequest<ReduceResult>(
            context,
            [operation = std::move(operation), elements](NativeMpiContext& native)
            {
                return detail::startReduce(
                    native,
                    operation.input,
                    operation.output,
                    operation.type,
                    operation.operation,
                    operation.root,
                    operation.communicator,
                    elements);
            },
            [elements](std::span<MPI_Status const>) { return ReduceResult{*elements}; },
            std::move(value),
            std::move(error),
            std::move(stopped),
            communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Gather operation,
        ValueCallback<GatherResult>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        auto const communicator = operation.communicator;
        submitRequest<GatherResult>(
            context,
            [operation = std::move(operation), resultBytes](NativeMpiContext& native)
            {
                return detail::startGather(
                    native,
                    operation.input,
                    operation.output,
                    operation.root,
                    operation.communicator,
                    resultBytes);
            },
            [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
            std::move(value),
            std::move(error),
            std::move(stopped),
            communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        GatherV operation,
        ValueCallback<GatherResult>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        auto const communicator = operation.communicator;
        submitRequest<GatherResult>(
            context,
            [operation = std::move(operation), resultBytes](NativeMpiContext& native)
            {
                return detail::startGatherV(
                    native,
                    operation.input,
                    operation.output,
                    operation.receiveBytes,
                    operation.displacements,
                    operation.root,
                    operation.communicator,
                    resultBytes);
            },
            [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
            std::move(value),
            std::move(error),
            std::move(stopped),
            communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Barrier operation,
        ValueCallback<void>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        submitRequest<void>(
            context,
            [operation](NativeMpiContext& native) { return detail::startBarrier(native, operation.communicator); },
            [](std::span<MPI_Status const>) {},
            std::move(value),
            std::move(error),
            std::move(stopped),
            operation.communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        CreateCartesian operation,
        ValueCallback<TopologySnapshot>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        submitBlocking<TopologySnapshot>(
            context,
            [operation = std::move(operation)](NativeMpiContext& native) mutable
            {
                return detail::createCartesian(
                    native,
                    std::move(operation.dimensions),
                    std::move(operation.periodic),
                    operation.worldSize,
                    operation.hostLocalRank);
            },
            std::move(value),
            std::move(error),
            std::move(stopped),
            worldCommunicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        DuplicateCommunicator operation,
        ValueCallback<CommunicatorId>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        submitBlocking<CommunicatorId>(
            context,
            [operation](NativeMpiContext& native)
            { return detail::duplicateCommunicator(native, operation.communicator); },
            std::move(value),
            std::move(error),
            std::move(stopped),
            operation.communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        SplitCommunicator operation,
        ValueCallback<std::optional<CommunicatorInfo>>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        submitBlocking<std::optional<CommunicatorInfo>>(
            context,
            [operation](NativeMpiContext& native)
            { return detail::splitCommunicator(native, operation.color, operation.key, operation.communicator); },
            std::move(value),
            std::move(error),
            std::move(stopped),
            operation.communicator);
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        DestroyCommunicator operation,
        ValueCallback<void>::type value,
        ErrorCallback error,
        StoppedCallback stopped)
    {
        submitBlocking<void>(
            context,
            [operation](NativeMpiContext& native) { detail::destroyCommunicator(native, operation.communicator); },
            std::move(value),
            std::move(error),
            std::move(stopped),
            operation.communicator);
    }

} // namespace caravan
