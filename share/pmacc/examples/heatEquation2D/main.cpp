/* Copyright 2023-2024 Tapish Narwal
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include "include/PngCreator.hpp"
#include "include/SetBoundaryConditions.hpp"
#include "include/StencilFourPoint.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/mpi/GatherSlice.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <span>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <caravan/mpi.hpp>

#define NUM_STEPS 1000
#define NUM_DEVICES_PER_DIM 2
#define THERMAL_DIFFUSIVITY 4 // POSITIVE FACTOR
#define DX 4 // GRID SPACING
#define DT 1 // TIME STEP - STABLE IF DT < (DX * DX) / (4 * THERMAL_DIFFUSIVITY)

template<typename T_Gather, typename T_GridBuffer, typename T_Queue>
inline auto createPng(
    uint32_t currentStep,
    T_Gather& gather,
    std::unique_ptr<T_GridBuffer> const& gridBuffer,
    caravan::ControlContext& asyncContext,
    T_Queue& queue)
{
    /* gather::operator() gathers all the buffers and assembles those to
     * a complete picture discarding the guards.
     */
    if(gather->isParticipating())
    {
        pmacc::SubGrid<DIM2> const& subGrid = pmacc::Environment<DIM2>::get().SubGrid();
        auto bufferLayout = gridBuffer->getGridLayout();
        auto localDataExtents = bufferLayout.sizeWithoutGuardND();
        auto view = std::make_unique<pmacc::DeviceBuffer<float, DIM2>>(
            gridBuffer->getDeviceBuffer(),
            localDataExtents,
            bufferLayout.guardSizeND());
        // create a contiguous buffer required for gathering the data
        auto dataWithoutGuard = std::make_unique<pmacc::HostBuffer<float, DIM2>>(localDataExtents);
        auto copy = caravan::alpaka::copy(
            queue,
            dataWithoutGuard->getOwnedAlpakaView(),
            view->getOwnedAlpakaView(),
            localDataExtents.toAlpakaMemVec());
        asyncContext.wait(asyncContext.spawn(std::move(copy)));
        auto picture = gather->gatherSliceExplicit(
            *dataWithoutGuard,
            subGrid.getGlobalDomain().size,
            subGrid.getLocalDomain().offset);
        PngCreator png;
        if(gather->isMaster())
            png(currentStep, picture->getDataBox(), picture->capacityND());
    }
}

auto run(caravan::MpiContext& mpi) -> int
{
    auto const devices = pmacc::DataSpace<DIM2>::create(NUM_DEVICES_PER_DIM);
    auto const periodic = pmacc::DataSpace<DIM2>::create(0);
    pmacc::Environment<DIM2>::get().initDevices(mpi, devices, periodic);

    /** define a gloabl grid */
    pmacc::DataSpace<DIM2> const gridSize{256u, 256u};

    auto& gc = pmacc::Environment<DIM2>::get().GridController();

    /** device local grid size */
    pmacc::DataSpace<DIM2> const localGridSize{gridSize / devices};

    pmacc::Environment<DIM2>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);

    /** Get reference to subGrid object, which holds local position, global
     *  position, and size information, as offset of local position wrt global
     *  position
     */
    auto const& subGrid = pmacc::Environment<DIM2>::get().SubGrid();

    /** define mapping description, this defines the supercell size */
    using MappingDesc = pmacc::MappingDescription<DIM2, pmacc::math::CT::Int<16, 16>>;

    /** adds guards to the dataspace
     *  here guard size is calulated by using the supercell size
     */
    pmacc::GridLayout<DIM2> layout{subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT()};

    /** mapping description
     *  takes in the grid layout - CELLS dataspace + guard, and num of SUPERCELLS in guard
     */
    pmacc::DataSpace<DIM2> guardingSuperCells{1, 1};
    auto mapping = std::make_unique<MappingDesc>(layout.sizeND(), guardingSuperCells);

    /** define grid buffers, two because we dont do in place writes */

    auto buff1 = std::make_unique<pmacc::GridBuffer<float, DIM2>>(layout);
    auto buff2 = std::make_unique<pmacc::GridBuffer<float, DIM2>>(layout);

    pmacc::DataSpace<DIM2> guardingCells{1, 1};

    // add stencil directions only add up, down, left, and right exchanges. dont need
    // diagonals both buffers need exchanges because later we will swap pointers
    // and do exhanges in an alternating fashion
    StencilFourPoint stencilKernel{};
    for(auto i : stencilKernel.stencilDirections)
    {
        buff1->addExchange(pmacc::GUARD, pmacc::Mask(i), guardingCells, 0u);
        buff2->addExchange(pmacc::GUARD, pmacc::Mask(i), guardingCells, 1u);
    }

    // define mappers which run over the certain area, with a supercell size
    pmacc::AreaMapping<pmacc::type::CORE, MappingDesc> coreMapper(*mapping);
    pmacc::AreaMapping<pmacc::type::BORDER, MappingDesc> borderMapper(*mapping);

    auto const device = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
    pmacc::ComputeDeviceQueue communicationQueue(device);
    pmacc::ComputeDeviceQueue computeQueue(device);
    caravan::ControlContext asyncContext;
    using SuperCell = typename MappingDesc::SuperCellSize;
    auto residualBuffer = std::make_unique<pmacc::HostDeviceBuffer<float, DIM1>>(pmacc::DataSpace<DIM1>::create(1));

    auto boundaryKernel
        = pmacc::lockstep::exec::kernel(SetBoundaryConditions{}).config(borderMapper.getGridDim(), SuperCell{});
    auto initialValues = caravan::alpaka::sequence(
        caravan::alpaka::sequence(
            caravan::alpaka::fill(computeQueue, buff1->getDeviceBuffer().getOwnedAlpakaView(), 0u),
            caravan::alpaka::fill(computeQueue, buff2->getDeviceBuffer().getOwnedAlpakaView(), 0u)),
        caravan::alpaka::fill(computeQueue, residualBuffer->getDeviceBuffer().getOwnedAlpakaView(), 0u));
    auto initialBoundaries = caravan::alpaka::sequence(
        caravan::alpaka::sequence(
            std::move(initialValues),
            boundaryKernel(
                computeQueue,
                caravan::retain(buff1->getDeviceBuffer().getDataBox(), buff1->getDeviceBuffer().getOwnedAlpakaView()),
                NUM_DEVICES_PER_DIM,
                gc.getPosition(),
                subGrid.getLocalDomain().offset,
                gridSize,
                borderMapper)),
        boundaryKernel(
            computeQueue,
            caravan::retain(buff2->getDeviceBuffer().getDataBox(), buff2->getDeviceBuffer().getOwnedAlpakaView()),
            NUM_DEVICES_PER_DIM,
            gc.getPosition(),
            subGrid.getLocalDomain().offset,
            gridSize,
            borderMapper));
    asyncContext.wait(asyncContext.spawn(std::move(initialBoundaries)));

    auto gather = std::make_unique<pmacc::mpi::GatherSlice>();
    createPng(0u, gather, buff1, asyncContext, computeQueue);

    float reducedResidual = 0.0f;
    bool const isReductionRoot = mpi.topology().rank == 0;

    for(uint32_t i = 0; i < NUM_STEPS; i++)
    {
        auto communication = buff1->spawnCommunication(asyncContext, communicationQueue);

        auto core = pmacc::lockstep::exec::kernel(StencilFourPoint{})
                        .config(coreMapper.getGridDim(), SuperCell{})(
                            computeQueue,
                            caravan::retain(
                                buff1->getDeviceBuffer().getDataBox(),
                                buff1->getDeviceBuffer().getOwnedAlpakaView()),
                            caravan::retain(
                                buff2->getDeviceBuffer().getDataBox(),
                                buff2->getDeviceBuffer().getOwnedAlpakaView()),
                            caravan::retain(
                                residualBuffer->getDeviceBuffer().getDataBox(),
                                residualBuffer->getDeviceBuffer().getOwnedAlpakaView()),
                            THERMAL_DIFFUSIVITY,
                            DX,
                            DT,
                            coreMapper);
        auto deviceStep
            = asyncContext.onControl(caravan::whenAll(std::move(core), caravan::asSender(std::move(communication))))
              | caravan::letValue(
                  [&,
                   readView = buff1->getDeviceBuffer().getOwnedAlpakaView(),
                   writeView = buff2->getDeviceBuffer().getOwnedAlpakaView(),
                   residualView = residualBuffer->getDeviceBuffer().getOwnedAlpakaView(),
                   residualHostView = residualBuffer->getHostBuffer().getOwnedAlpakaView()]() mutable
                  {
                      auto boundary = boundaryKernel(
                          computeQueue,
                          caravan::retain(buff1->getDeviceBuffer().getDataBox(), readView),
                          NUM_DEVICES_PER_DIM,
                          gc.getPosition(),
                          subGrid.getLocalDomain().offset,
                          gridSize,
                          borderMapper);
                      auto border
                          = pmacc::lockstep::exec::kernel(StencilFourPoint{})
                                .config(borderMapper.getGridDim(), SuperCell{})(
                                    computeQueue,
                                    caravan::retain(buff1->getDeviceBuffer().getDataBox(), readView),
                                    caravan::retain(buff2->getDeviceBuffer().getDataBox(), writeView),
                                    caravan::retain(residualBuffer->getDeviceBuffer().getDataBox(), residualView),
                                    THERMAL_DIFFUSIVITY,
                                    DX,
                                    DT,
                                    borderMapper);
                      auto copyResidual = caravan::alpaka::copy(
                          computeQueue,
                          std::move(residualHostView),
                          residualView,
                          pmacc::DataSpace<DIM1>::create(1).toAlpakaMemVec());
                      auto resetResidual = caravan::alpaka::fill(computeQueue, std::move(residualView), 0u);
                      return caravan::alpaka::sequence(
                          caravan::alpaka::sequence(
                              caravan::alpaka::sequence(std::move(boundary), std::move(border)),
                              std::move(copyResidual)),
                          std::move(resetResidual));
                  });
        auto step = asyncContext.onControl(std::move(deviceStep))
                    | caravan::letValue(
                        [&]
                        {
                            return caravan::mpi::reduce(
                                mpi,
                                caravan::retain(
                                    std::as_bytes(std::span{residualBuffer->getHostBuffer().data(), 1}),
                                    std::shared_ptr<void>{}),
                                caravan::retain(
                                    std::as_writable_bytes(std::span{&reducedResidual, 1}),
                                    std::shared_ptr<void>{}),
                                caravan::ScalarType::float32,
                                caravan::ReduceOperation::sum,
                                caravan::Peer{0});
                        });
        asyncContext.wait(asyncContext.spawn(std::move(step)));

        std::swap(buff1, buff2);
        createPng(i + 1u, gather, buff1, asyncContext, computeQueue);
        if(isReductionRoot)
            std::cout << "Residual at time " << DT * i << " = " << reducedResidual << std::endl;
    }

    gather.reset();
    pmacc::Environment<DIM2>::get().finalize();

    return 0;
}

auto main(int argc, char** argv) -> int
{
    return caravan::MpiRuntime::run(argc, argv, [](caravan::MpiContext& mpi) { return run(mpi); });
}
