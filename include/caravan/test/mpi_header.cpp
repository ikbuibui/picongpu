/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/mpi.hpp>

#ifdef MPI_VERSION
#    error "caravan/mpi.hpp must not expose mpi.h"
#endif

using Context = caravan::MpiContext;
using ConstBuffer = caravan::ConstBufferLease;
using Buffer = caravan::BufferLease;

static_assert(std::is_same_v<decltype(std::declval<ConstBuffer>().data()), void const*>);
static_assert(std::is_same_v<decltype(std::declval<Buffer>().data()), void*>);
static_assert(caravan::Sender<decltype(caravan::mpi::send(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  caravan::Peer{},
                  caravan::MessageTag{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::receive(
                  std::declval<Context&>(),
                  std::declval<Buffer>(),
                  caravan::Peer{},
                  caravan::MessageTag{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::allReduce(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  caravan::ScalarType::int32,
                  caravan::ReduceOperation::sum))>);
static_assert(caravan::Sender<decltype(caravan::mpi::reduce(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  caravan::ScalarType::int32,
                  caravan::ReduceOperation::sum,
                  caravan::Peer{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::gather(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  caravan::Peer{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::gatherV(
                  std::declval<Context&>(),
                  std::declval<ConstBuffer>(),
                  std::declval<Buffer>(),
                  std::vector<std::size_t>{},
                  std::vector<std::size_t>{},
                  caravan::Peer{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::barrier(std::declval<Context&>()))>);
static_assert(
    caravan::Sender<
        decltype(caravan::mpi::createCartesian(std::declval<Context&>(), std::vector<int>{}, std::vector<bool>{}))>);
static_assert(caravan::Sender<decltype(caravan::mpi::duplicateCommunicator(std::declval<Context&>()))>);
static_assert(
    caravan::Sender<decltype(caravan::mpi::splitCommunicator(std::declval<Context&>(), std::optional<int>{}, 0))>);
static_assert(caravan::Sender<
              decltype(caravan::mpi::destroyCommunicator(std::declval<Context&>(), caravan::worldCommunicator))>);

int main()
{
}
