/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/communication/manager_common.hpp"
#include "pmacc/mpi/GetMPI_Op.hpp"
#include "pmacc/mpi/GetMPI_StructAsArray.hpp"
#include "pmacc/mpi/reduceMethods/AllReduce.hpp"
#include "pmacc/types.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <caravan/core.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace pmacc
{
    namespace mpi
    {
        /** reduce data over selected mpi ranks */
        struct MPIReduce
        {
            MPIReduce() = default;

            virtual ~MPIReduce()
            {
                if(mpiContext && caravanCommunicator)
                {
                    try
                    {
                        caravan::syncWait(
                            caravan::mpi::destroyCommunicator(*mpiContext, caravanCommunicator->communicator));
                    }
                    catch(std::exception const& error)
                    {
                        std::cerr << "Failed to destroy Caravan reduction communicator: " << error.what() << '\n';
                    }
                    catch(...)
                    {
                        std::cerr << "Failed to destroy Caravan reduction communicator\n";
                    }
                }
                else if(comm != MPI_COMM_NULL)
                    MPI_CHECK_NO_EXCEPT(MPI_Comm_free(&comm));
            }

            /* defines if the result of the MPI operation is valid
             *
             * @tparam MPIMethod type of the reduction method
             * @param method used reduction method e.g.,
             *                reduceMethods::AllReduce, reduceMethods::Reduce
             * @return if resut of operator() is valid*/
            template<class MPIMethod>
            bool hasResult(MPIMethod const& method)
            {
                if(!isMPICommInitialized)
                    participate(true);
                return method.hasResult(mpiRank);
            }

            /** defines if the result of the MPI operation is valid
             *
             * The reduction method reduceMethods::AllReduce is used.
             *
             * @return if result of operator() is valid
             */
            bool hasResult()
            {
                if(!isMPICommInitialized)
                    participate(true);
                return this->hasResult(::pmacc::mpi::reduceMethods::AllReduce());
            }

            /* Activate participation for reduce algorithm.
             * Must called from any mpi process. This function use global blocking mpi calls.
             * @param isActive true if mpi rank should be part of reduce operation, else false
             */
            void participate(bool isActive)
            {
                if(mpiContext && caravanCommunicator)
                    caravan::syncWait(
                        caravan::mpi::destroyCommunicator(*mpiContext, caravanCommunicator->communicator));
                else if(comm != MPI_COMM_NULL)
                    MPI_CHECK(MPI_Comm_free(&comm));

                mpiRank = -1;
                numRanks = 0;
                caravanCommunicator.reset();
                comm = MPI_COMM_NULL;
                mpiContext = Environment<>::get().getMpiContext();

                if(mpiContext)
                {
                    auto const world = mpiContext->topology();
                    caravanCommunicator
                        = caravan::syncWait<std::optional<caravan::CommunicatorInfo>>(caravan::mpi::splitCommunicator(
                            *mpiContext,
                            isActive ? std::optional<int>{0} : std::nullopt,
                            world.rank));
                    if(caravanCommunicator)
                    {
                        mpiRank = caravanCommunicator->rank;
                        numRanks = caravanCommunicator->size;
                    }
                    isMPICommInitialized = true;
                    return;
                }

                int countRanks;
                MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &countRanks));
                std::vector<int> reduceRank(countRanks);
                std::vector<int> groupRanks(countRanks);
                MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));

                if(!isActive)
                    mpiRank = -1;

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(&mpiRank, 1, MPI_INT, &reduceRank[0], 1, MPI_INT, MPI_COMM_WORLD));

                for(int i = 0; i < countRanks; ++i)
                {
                    if(reduceRank[i] != -1)
                        groupRanks[numRanks++] = reduceRank[i];
                }

                MPI_Group group = MPI_GROUP_NULL;
                MPI_Group newgroup = MPI_GROUP_NULL;
                MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
                MPI_CHECK(MPI_Group_incl(group, numRanks, &groupRanks[0], &newgroup));
                MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &comm));

                if(mpiRank != -1)
                    MPI_CHECK(MPI_Comm_rank(comm, &mpiRank));
                MPI_CHECK(MPI_Group_free(&group));
                MPI_CHECK(MPI_Group_free(&newgroup));
                isMPICommInitialized = true;
            }

            /* Reduce elements on cpu memory
             * call hasResult to see if returned value is valid
             *
             * @param func binary functor for reduce which takes two arguments, first argument is the source and get
             * the new reduced value. Functor must specialize the function getMPI_Op.
             * @param dest buffer for result data
             * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one
             * dimension access)
             * @param n number of elements to reduce
             * @param method mpi method for reduce
             *
             */
            template<class Functor, typename Type, class ReduceMethod>
            HINLINE void operator()(Functor func, Type* dest, Type* src, size_t const n, ReduceMethod const method)
            {
                if(!isMPICommInitialized)
                    participate(true);
                using ValueType = Type;

                if(mpiContext)
                {
                    if(!caravanCommunicator)
                        throw std::logic_error("Inactive rank cannot submit an MPI reduction");
                    eventSystem::getTransactionEvent().waitForFinished();
                    caravan::syncWait(
                        caravan::mpi::request<void>(
                            *mpiContext,
                            [=, communicator = caravanCommunicator->communicator](caravan::NativeMpiContext& context)
                            {
                                auto const descriptor = ::pmacc::mpi::getMPI_StructAsArray<ValueType>();
                                auto const elements = n * descriptor.sizeMultiplier;
                                caravan::NativeRequestBatch batch({MPI_REQUEST_NULL});
                                int error;
                                if constexpr(std::is_same_v<
                                                 std::remove_cvref_t<ReduceMethod>,
                                                 reduceMethods::AllReduce>)
                                    error = MPI_Iallreduce(
                                        src,
                                        dest,
                                        static_cast<int>(elements),
                                        descriptor.dataType,
                                        ::pmacc::mpi::getMPI_Op<Functor>(),
                                        context.communicator(communicator),
                                        &batch.requests[0]);
                                else
                                    error = MPI_Ireduce(
                                        src,
                                        dest,
                                        static_cast<int>(elements),
                                        descriptor.dataType,
                                        ::pmacc::mpi::getMPI_Op<Functor>(),
                                        0,
                                        context.communicator(communicator),
                                        &batch.requests[0]);
                                if(error != MPI_SUCCESS)
                                    throw std::runtime_error("PMacc native MPI reduction start failed");
                                return batch;
                            },
                            [](std::span<MPI_Status const>) {},
                            caravanCommunicator->communicator));
                    return;
                }

                method(
                    func,
                    dest,
                    src,
                    n * ::pmacc::mpi::getMPI_StructAsArray<ValueType>().sizeMultiplier,
                    ::pmacc::mpi::getMPI_StructAsArray<ValueType>().dataType,
                    ::pmacc::mpi::getMPI_Op<Functor>(),
                    comm);
            }

            /* Reduce elements on cpu memory
             * the default reduce method is allReduce which means that any host get the reduced value back
             *
             * @param func binary functor for reduce which takes two arguments, first argument is the source and get
             * the new reduced value. Functor must specialize the function getMPI_Op.
             * @param dest buffer for result data
             * @param src a class or a pointer where the reduce algorithm can access the value by operator [] (one
             * dimension access)
             * @param n number of elements to reduce
             *
             * @return reduced value
             */
            template<class Functor, typename Type>
            HINLINE void operator()(Functor func, Type* dest, Type* src, size_t const n)
            {
                if(!isMPICommInitialized)
                    participate(true);
                this->operator()(func, dest, src, n, ::pmacc::mpi::reduceMethods::AllReduce());
            }


        private:
            MPI_Comm comm{MPI_COMM_NULL};
            std::optional<caravan::CommunicatorInfo> caravanCommunicator;
            caravan::MpiContext* mpiContext{nullptr};
            int mpiRank{-1};
            int numRanks{0};
            bool isMPICommInitialized{false};
        };
    } // namespace mpi
} // namespace pmacc
