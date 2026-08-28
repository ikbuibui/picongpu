/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <climits>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <caravan/mpi/error.hpp>
#include <caravan/mpi/native.hpp>
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

    Future<AllReduceResult> MpiContext::allReduce(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator)
    {
        auto elements = std::make_shared<std::size_t>(0u);
        return nativeFuture<AllReduceResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input), output = std::move(output), type, operation, communicator, elements](
                NativeMpiContext& context)
            { return detail::startAllReduce(context, input, output, type, operation, communicator, elements); },
            [elements](std::span<MPI_Status const>) { return AllReduceResult{*elements}; },
            communicator);
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

    Future<ReduceResult> MpiContext::reduce(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator)
    {
        auto elements = std::make_shared<std::size_t>(0u);
        return nativeFuture<ReduceResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input), output = std::move(output), type, operation, root, communicator, elements](
                NativeMpiContext& context)
            { return detail::startReduce(context, input, output, type, operation, root, communicator, elements); },
            [elements](std::span<MPI_Status const>) { return ReduceResult{*elements}; },
            communicator);
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

    Future<GatherResult> MpiContext::gather(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        Peer root,
        CommunicatorId communicator)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        return nativeFuture<GatherResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input), output = std::move(output), root, communicator, resultBytes](
                NativeMpiContext& context)
            { return detail::startGather(context, input, output, root, communicator, resultBytes); },
            [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
            communicator);
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

    Future<GatherResult> MpiContext::gatherV(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        std::vector<std::size_t> receiveBytes,
        std::vector<std::size_t> displacements,
        Peer root,
        CommunicatorId communicator)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        return nativeFuture<GatherResult>(
            *this,
            std::move(dataReady),
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

    NativeRequestBatch detail::startBarrier(NativeMpiContext& context, CommunicatorId communicator)
    {
        NativeRequestBatch batch({MPI_REQUEST_NULL});
        int const error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ibarrier", error);
        return batch;
    }

    Event MpiContext::barrier(Event predecessor, CommunicatorId communicator)
    {
        return nativeEvent(
            *this,
            std::move(predecessor),
            [communicator](NativeMpiContext& context) { return detail::startBarrier(context, communicator); },
            [](std::span<MPI_Status const>) {},
            communicator);
    }

} // namespace caravan
