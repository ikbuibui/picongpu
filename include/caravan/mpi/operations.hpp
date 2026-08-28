/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <caravan/mpi/native.hpp>

namespace caravan
{
    namespace mpi
    {
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
} // namespace caravan
