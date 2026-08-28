/* Copyright 2023-2024 Rene Widera
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
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

#include <array>
#include <optional>
#include <vector>

#include <caravan/core.hpp>
#include <caravan/mpi.hpp>
#include <mpi.h>

namespace pmacc
{
    namespace mpi
    {
        //! Gather data of a 2D Cartesian host buffer into a single MPI rank's host memory.
        class GatherSlice
        {
        private:
            MPI_Comm gatherComm = MPI_COMM_NULL;
            std::optional<caravan::CommunicatorInfo> caravanGatherComm;
            caravan::MpiContext* mpiContext = nullptr;
            // gather rank zero will hold final data
            int gatherRank = -1;
            // number of ranks participating in the gather operation
            int numRanksInPlane = 0;

        public:
            GatherSlice()
            {
            }

            virtual ~GatherSlice()
            {
                if(mpiContext && caravanGatherComm)
                {
                    try
                    {
                        caravan::syncWait(
                            caravan::mpi::destroyCommunicator(*mpiContext, caravanGatherComm->communicator));
                    }
                    catch(std::exception const& error)
                    {
                        std::cerr << "Failed to destroy Caravan gather communicator: " << error.what() << '\n';
                    }
                }
                else if(gatherComm != MPI_COMM_NULL)
                {
                    auto err = MPI_Comm_free(&gatherComm);
                    if(err != MPI_SUCCESS)
                        std::cerr << __FILE__ << ":" << __LINE__ << "MPI_Comm_free failed." << std::endl;
                }
            }

            /** Check if MPI rank is the gather master rank.
             *
             * The master will return the data when calling gatherSlice().
             *
             * @return True if this MPI rank is returning the gathered data during gatherSlice() operation, else false.
             */
            bool isMaster() const
            {
                return gatherRank == 0;
            }

            /** Check if this MPI rank gathers the data.
             *
             * @return True if this MPI rank returns the gathered data during gatherSlice() operation, else false.
             */
            bool hasResult() const
            {
                return isMaster();
            }

            /** Query if MPI rank is part of the gather group.
             *
             * @return True if MPI rank is taking part on the gather operation, else false.
             */
            bool isParticipating() const
            {
                return gatherRank != -1;
            }

            /** Announce participation of the MPI rank in the gather operation
             *
             * @attention Must be called from all MPI ranks even if they do not participate.
             *
             * @param isActive True if MPI rank has data to gather, else false.
             * @return If the caller will contain the gathered data. @see isMaster()
             */
            bool participate(bool isActive)
            {
                mpiContext = Environment<>::get().getMpiContext();
                if(mpiContext)
                {
                    auto const world = mpiContext->topology();
                    caravanGatherComm
                        = caravan::syncWait<std::optional<caravan::CommunicatorInfo>>(caravan::mpi::splitCommunicator(
                            *mpiContext,
                            isActive ? std::optional<int>{0} : std::nullopt,
                            world.rank));
                    gatherRank = caravanGatherComm ? caravanGatherComm->rank : -1;
                    numRanksInPlane = caravanGatherComm ? caravanGatherComm->size : 0;
                    return isMaster();
                }

                int countRanks;
                int globalMpiRank;
                MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &countRanks));
                std::vector<int> allRank(countRanks);
                std::vector<int> groupRanks(countRanks);
                MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &globalMpiRank));

                if(!isActive)
                    globalMpiRank = -1;

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                MPI_CHECK(MPI_Allgather(&globalMpiRank, 1, MPI_INT, allRank.data(), 1, MPI_INT, MPI_COMM_WORLD));

                int numRanks = 0;
                for(int i = 0; i < countRanks; ++i)
                {
                    if(allRank[i] != -1)
                    {
                        groupRanks[numRanks] = allRank[i];
                        numRanks++;
                    }
                }
                numRanksInPlane = numRanks;

                MPI_Group group = MPI_GROUP_NULL;
                MPI_Group newgroup = MPI_GROUP_NULL;
                MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
                MPI_CHECK(MPI_Group_incl(group, numRanks, groupRanks.data(), &newgroup));

                MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &gatherComm));

                if(globalMpiRank != -1)
                {
                    MPI_CHECK(MPI_Comm_rank(gatherComm, &gatherRank));
                }
                MPI_CHECK(MPI_Group_free(&group));
                MPI_CHECK(MPI_Group_free(&newgroup));

                return this->isMaster();
            }

            /** gather data
             *
             * Must be called by all participating MPI ranks.
             * If a non-participating MPI rank is calling the method the returned buffer will be empty.
             * @attention The master rank will allocate host memory for the received data.
             *
             * @tparam T_DataType Slice buffer data type.
             * @param localInputSlice Buffer with local slice data. Buffer memory must be contiguous without line
             * paddings. Buffer extents can be different for each MPI rank.
             * @param globalSliceExtent extent in elements of the global slice
             * @param localSliceOffset local offset in elements relative to the global slice origin
             * @return shared pointer to host buffer with gathered slice data (only master has valid data)
             */
            template<typename T_DataType>
            auto gatherSlice(
                HostBuffer<T_DataType, DIM2>& localInputSlice,
                DataSpace<DIM2> globalSliceExtent,
                DataSpace<DIM2> localSliceOffset) const
            {
                using ValueType = T_DataType;
                // Guard against wrong usage, only MPI ranks which are participating into the gather are allowed to
                // call corresponding MPI functions.
                if(!isParticipating())
                    return std::shared_ptr<HostBuffer<ValueType, DIM2>>{};

                // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                eventSystem::getTransactionEvent().waitForFinished();
                // get number of elements per participating mpi rank
                auto extentPerDevice = std::vector<DataSpace<DIM2>>(numRanksInPlane);
                auto offsetPerDevice = std::vector<DataSpace<DIM2>>(numRanksInPlane);
                auto localSliceSize = localInputSlice.capacityND();

                if(mpiContext)
                {
                    std::array<int, 2> localExtent{localSliceSize.x(), localSliceSize.y()};
                    std::array<int, 2> localOffset{localSliceOffset.x(), localSliceOffset.y()};
                    std::vector<std::array<int, 2>> extents(numRanksInPlane);
                    std::vector<std::array<int, 2>> offsets(numRanksInPlane);
                    caravan::syncWait<caravan::GatherResult>(caravan::mpi::gather(
                        *mpiContext,
                        caravan::BufferLease::borrowed(localExtent.data(), sizeof(localExtent)),
                        caravan::BufferLease::borrowed(extents.data(), extents.size() * sizeof(extents[0])),
                        caravan::Peer{0},
                        caravanGatherComm->communicator));
                    caravan::syncWait<caravan::GatherResult>(caravan::mpi::gather(
                        *mpiContext,
                        caravan::BufferLease::borrowed(localOffset.data(), sizeof(localOffset)),
                        caravan::BufferLease::borrowed(offsets.data(), offsets.size() * sizeof(offsets[0])),
                        caravan::Peer{0},
                        caravanGatherComm->communicator));
                    if(isMaster())
                        for(int rank = 0; rank < numRanksInPlane; ++rank)
                        {
                            extentPerDevice[rank] = DataSpace<DIM2>(extents[rank][0], extents[rank][1]);
                            offsetPerDevice[rank] = DataSpace<DIM2>(offsets[rank][0], offsets[rank][1]);
                        }
                }
                else
                {
                    MPI_CHECK(MPI_Gather(
                        reinterpret_cast<int*>(&localSliceSize),
                        2,
                        MPI_INT,
                        reinterpret_cast<int*>(extentPerDevice.data()),
                        2,
                        MPI_INT,
                        0,
                        gatherComm));
                    MPI_CHECK(MPI_Gather(
                        reinterpret_cast<int*>(&localSliceOffset),
                        2,
                        MPI_INT,
                        reinterpret_cast<int*>(offsetPerDevice.data()),
                        2,
                        MPI_INT,
                        0,
                        gatherComm));
                }

                std::vector<int> displs(numRanksInPlane);
                std::vector<int> count(numRanksInPlane);

                int offset = 0;
                int globalNumElements = 0u;

                if(isMaster())
                {
                    //! @todo replace by std::scan
                    for(int i = 0; i < numRanksInPlane; ++i)
                    {
                        displs[i] = offset * sizeof(ValueType);
                        count[i] = extentPerDevice[i].productOfComponents() * sizeof(ValueType);
                        offset += extentPerDevice[i].productOfComponents();
                        globalNumElements += extentPerDevice[i].productOfComponents();
                    }
                }

                // gather all data from other ranks
                auto allData = std::vector<ValueType>(globalNumElements);
                int localNumElements = localSliceSize.productOfComponents();

                if(mpiContext)
                {
                    std::vector<std::size_t> receiveBytes(count.begin(), count.end());
                    std::vector<std::size_t> displacements(displs.begin(), displs.end());
                    caravan::syncWait<caravan::GatherResult>(caravan::mpi::gatherV(
                        *mpiContext,
                        caravan::BufferLease::borrowed(
                            localInputSlice.data(),
                            static_cast<std::size_t>(localNumElements) * sizeof(ValueType)),
                        caravan::BufferLease::borrowed(allData.data(), allData.size() * sizeof(ValueType)),
                        std::move(receiveBytes),
                        std::move(displacements),
                        caravan::Peer{0},
                        caravanGatherComm->communicator));
                }
                else
                    MPI_CHECK(MPI_Gatherv(
                        reinterpret_cast<char*>(localInputSlice.data()),
                        localNumElements * sizeof(ValueType),
                        MPI_CHAR,
                        reinterpret_cast<char*>(allData.data()),
                        count.data(),
                        displs.data(),
                        MPI_CHAR,
                        0,
                        gatherComm));

                std::shared_ptr<HostBuffer<ValueType, DIM2>> globalField;
                if(isMaster())
                {
                    // globalNumElements is only on the master rank valid
                    PMACC_VERIFY_MSG(
                        globalSliceExtent.productOfComponents() == globalNumElements,
                        "Expected and gathered number of elements differ.");

                    globalField = std::make_shared<HostBuffer<ValueType, DIM2>>(globalSliceExtent);
                    auto globalFieldBox = globalField->getDataBox();

                    // aggregate data of all MPI ranks into a single 2D buffer
                    for(int dataSetNumber = 0; dataSetNumber < numRanksInPlane; ++dataSetNumber)
                    {
                        for(int y = 0; y < extentPerDevice[dataSetNumber].y(); ++y)
                            for(int x = 0; x < extentPerDevice[dataSetNumber].x(); ++x)
                            {
                                globalFieldBox(DataSpace<DIM2>(x, y) + offsetPerDevice[dataSetNumber]) = allData
                                    [displs[dataSetNumber] / sizeof(ValueType) + y * extentPerDevice[dataSetNumber].x()
                                     + x];
                            }
                    }
                }
                return globalField;
            }
        };
    } // namespace mpi
} // namespace pmacc
