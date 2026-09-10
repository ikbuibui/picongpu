/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <climits>
#include <cstddef>
#include <cstdint>
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
        template<typename T_Buffer>
        bool validBuffer(T_Buffer const& buffer)
        {
            return buffer.value.size_bytes() <= static_cast<std::size_t>(INT_MAX);
        }

        bool buffersOverlap(ConstMpiBuffer const& input, MpiBuffer const& output) noexcept
        {
            if(input.value.size_bytes() == 0u || output.value.size_bytes() == 0u)
                return false;
            auto const inputAddress = reinterpret_cast<std::uintptr_t>(input.value.data());
            auto const outputAddress = reinterpret_cast<std::uintptr_t>(output.value.data());
            return inputAddress <= outputAddress ? outputAddress - inputAddress < input.value.size_bytes()
                                                 : inputAddress - outputAddress < output.value.size_bytes();
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
            mpi::operation_detail::ErrorCallback error)
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
                    std::move(error)});
        }

        template<typename T, typename T_Operation>
        void submitInvocation(
            MpiContext& context,
            T_Operation operation,
            typename mpi::operation_detail::ValueCallback<T>::type value,
            mpi::operation_detail::ErrorCallback error)
        {
            detail::NativeAccess::invoke(
                context,
                detail::NativeInvocation{
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
                    std::move(error)});
        }
    } // namespace

    NativeRequestBatch detail::startSend(
        NativeMpiContext& context,
        ConstMpiBuffer const& buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || destination.any || destination.value < 0 || tag.any || tag.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI send");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.owner});
        int const error = MPI_Isend(
            buffer.value.data(),
            static_cast<int>(buffer.value.size_bytes()),
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
        MpiBuffer const& buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || (!source.any && source.value < 0) || (!tag.any && tag.value < 0))
            throw std::invalid_argument("Invalid Caravan MPI receive");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.owner});
        int const error = MPI_Irecv(
            buffer.value.data(),
            static_cast<int>(buffer.value.size_bytes()),
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
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& elements)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output)
           || input.value.size_bytes() % elementBytes != 0u || output.value.size_bytes() < input.value.size_bytes())
            throw std::invalid_argument("Invalid Caravan MPI all-reduce");
        *elements = input.value.size_bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner, elements});
        void const* const sendBuffer = input.value.data() == output.value.data() ? MPI_IN_PLACE : input.value.data();
        int const error = MPI_Iallreduce(
            sendBuffer,
            output.value.data(),
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
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& elements)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output)
           || input.value.size_bytes() % elementBytes != 0u || output.value.size_bytes() < input.value.size_bytes()
           || input.value.data() == output.value.data() || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI reduce");
        *elements = input.value.size_bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner, elements});
        int const error = MPI_Ireduce(
            input.value.data(),
            output.value.data(),
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
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || buffersOverlap(input, output) || root.any || root.value < 0)
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
            if(size <= 0 || input.value.size_bytes() > output.value.size_bytes() / static_cast<std::size_t>(size))
                throw std::invalid_argument("Caravan MPI gather output is too small");
            *resultBytes = input.value.size_bytes() * static_cast<std::size_t>(size);
        }

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner, resultBytes});
        error = MPI_Igather(
            input.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            output.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igather", error);
        return batch;
    }

    NativeRequestBatch detail::startAllGather(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || buffersOverlap(input, output))
            throw std::invalid_argument("Invalid Caravan MPI all-gather");

        auto const native = context.communicator(communicator);
        int size = 0;
        int error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI all-gather communicator query", error);
        if(size <= 0 || input.value.size_bytes() > output.value.size_bytes() / static_cast<std::size_t>(size))
            throw std::invalid_argument("Caravan MPI all-gather output is too small");
        *resultBytes = input.value.size_bytes() * static_cast<std::size_t>(size);

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner, resultBytes});
        error = MPI_Iallgather(
            input.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            output.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Iallgather", error);
        return batch;
    }

    NativeRequestBatch detail::startGatherV(
        NativeMpiContext& context,
        ConstMpiBuffer const& input,
        MpiBuffer const& output,
        std::vector<std::size_t> const& receiveBytes,
        std::vector<std::size_t> const& displacements,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || buffersOverlap(input, output)
           || receiveBytes.size() != displacements.size() || root.any || root.value < 0)
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
                if(end > output.value.size_bytes())
                    throw std::invalid_argument("Caravan MPI variable gather output is too small");
                *resultBytes += static_cast<std::size_t>((*counts)[i]);
            }
        }

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.owner, output.owner, counts, offsets, resultBytes});
        error = MPI_Igatherv(
            input.value.data(),
            static_cast<int>(input.value.size_bytes()),
            MPI_BYTE,
            output.value.data(),
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

    mpi::OperationSender<mpi::operation_detail::Send> mpi::send(
        MpiContext& context,
        ConstMpiBuffer buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        return {context, operation_detail::Send{std::move(buffer), destination, tag, communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::Receive> mpi::receive(
        MpiContext& context,
        MpiBuffer buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        return {context, operation_detail::Receive{std::move(buffer), source, tag, communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::AllReduce> mpi::allReduce(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator)
    {
        return {
            context,
            operation_detail::AllReduce{std::move(input), std::move(output), type, operation, communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::Reduce> mpi::reduce(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator)
    {
        return {
            context,
            operation_detail::Reduce{std::move(input), std::move(output), type, operation, root, communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::Gather> mpi::gather(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        Peer root,
        CommunicatorId communicator)
    {
        return {context, operation_detail::Gather{std::move(input), std::move(output), root, communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::AllGather> mpi::allGather(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
        CommunicatorId communicator)
    {
        return {context, operation_detail::AllGather{std::move(input), std::move(output), communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::GatherV> mpi::gatherV(
        MpiContext& context,
        ConstMpiBuffer input,
        MpiBuffer output,
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

    mpi::OperationSender<mpi::operation_detail::Barrier> mpi::barrier(MpiContext& context, CommunicatorId communicator)
    {
        return {context, operation_detail::Barrier{communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::CreateCartesian> mpi::createCartesian(
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

    mpi::OperationSender<mpi::operation_detail::DuplicateCommunicator> mpi::duplicateCommunicator(
        MpiContext& context,
        CommunicatorId communicator)
    {
        return {context, operation_detail::DuplicateCommunicator{communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::SplitCommunicator> mpi::splitCommunicator(
        MpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator)
    {
        return {context, operation_detail::SplitCommunicator{color, key, communicator}};
    }

    mpi::OperationSender<mpi::operation_detail::DestroyCommunicator> mpi::destroyCommunicator(
        MpiContext& context,
        CommunicatorId communicator)
    {
        return {context, operation_detail::DestroyCommunicator{communicator}};
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Send operation,
        ValueCallback<SendResult>::type value,
        ErrorCallback error)
    {
        auto const bytes = operation.buffer.value.size_bytes();
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Receive operation,
        ValueCallback<ReceiveResult>::type value,
        ErrorCallback error)
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        AllReduce operation,
        ValueCallback<AllReduceResult>::type value,
        ErrorCallback error)
    {
        auto elements = std::make_shared<std::size_t>(0u);
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Reduce operation,
        ValueCallback<ReduceResult>::type value,
        ErrorCallback error)
    {
        auto elements = std::make_shared<std::size_t>(0u);
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Gather operation,
        ValueCallback<GatherResult>::type value,
        ErrorCallback error)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        AllGather operation,
        ValueCallback<GatherResult>::type value,
        ErrorCallback error)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        submitRequest<GatherResult>(
            context,
            [operation = std::move(operation), resultBytes](NativeMpiContext& native)
            {
                return detail::startAllGather(
                    native,
                    operation.input,
                    operation.output,
                    operation.communicator,
                    resultBytes);
            },
            [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
            std::move(value),
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        GatherV operation,
        ValueCallback<GatherResult>::type value,
        ErrorCallback error)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        Barrier operation,
        ValueCallback<void>::type value,
        ErrorCallback error)
    {
        submitRequest<void>(
            context,
            [operation](NativeMpiContext& native) { return detail::startBarrier(native, operation.communicator); },
            [](std::span<MPI_Status const>) {},
            std::move(value),
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        CreateCartesian operation,
        ValueCallback<TopologySnapshot>::type value,
        ErrorCallback error)
    {
        submitInvocation<TopologySnapshot>(
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
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        DuplicateCommunicator operation,
        ValueCallback<CommunicatorId>::type value,
        ErrorCallback error)
    {
        submitInvocation<CommunicatorId>(
            context,
            [operation](NativeMpiContext& native)
            { return detail::duplicateCommunicator(native, operation.communicator); },
            std::move(value),
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        SplitCommunicator operation,
        ValueCallback<std::optional<CommunicatorInfo>>::type value,
        ErrorCallback error)
    {
        submitInvocation<std::optional<CommunicatorInfo>>(
            context,
            [operation](NativeMpiContext& native)
            { return detail::splitCommunicator(native, operation.color, operation.key, operation.communicator); },
            std::move(value),
            std::move(error));
    }

    void mpi::operation_detail::submit(
        MpiContext& context,
        DestroyCommunicator operation,
        ValueCallback<void>::type value,
        ErrorCallback error)
    {
        submitInvocation<void>(
            context,
            [operation](NativeMpiContext& native) { detail::destroyCommunicator(native, operation.communicator); },
            std::move(value),
            std::move(error));
    }

} // namespace caravan
