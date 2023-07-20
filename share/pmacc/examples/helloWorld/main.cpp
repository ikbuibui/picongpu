/* Copyright 2013-2022 Rene Widera
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/math/Vector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"

#include "HelloKernel.hpp"

#include <iostream>
#include <fstream>

auto main(int argc, char **argv) -> int
{
    const pmacc::DataSpace<DIM2> devices = pmacc::DataSpace<DIM2>::create(2);
    const pmacc::DataSpace<DIM2> periodic = pmacc::DataSpace<DIM2>::create(0);
    pmacc::Environment<DIM2>::get().initDevices(devices, periodic);

    /* define a grid */
    const pmacc::DataSpace<DIM2> gridSize{100u, 100u};

    pmacc::GridController<DIM2> &gc = pmacc::Environment<DIM2>::get().GridController();

    /* device local grid size */
    const pmacc::DataSpace<DIM2> localGridSize{gridSize / devices};

    pmacc::Environment<DIM2>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);

    /* Get reference to subGrid object, which holds local position, global position,
     * and size information, as offset of local position wrt global position 
     */
    const pmacc::SubGrid<DIM2> &subGrid = pmacc::Environment<DIM2>::get().SubGrid();

    /* define mapping description, this takes in guard size? (here 4x4 cells in 1 supercell, which is a guard) 
     * so i think supercell size is described here, but cant i decouple supercell size from guard size?
     * maybe this isnt guard size, only supercell size
     */
    using MappingDesc = pmacc::MappingDescription<DIM2, pmacc::math::CT::Int<4, 4>>;

    /* adds guards to the dataspace 
     * here guard size is calulated by using the supercell size
     */
    pmacc::GridLayout<DIM2> layout{subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT()};
    std::cout << gc.getPosition() << "\t" << gc.getGlobalRank() << "\t" << subGrid.getLocalDomain().toString() << "\n";

    /* mapping 
     * takes in the grid layout - CELLS dataspace + guard, and num of SUPERCELLS in guard 
     */
    std::unique_ptr<MappingDesc> mapping;
    mapping = std::make_unique<MappingDesc>(layout.getDataSpace(), pmacc::DataSpace<DIM2>::create(1));

    /* mapping the supercells to the actual local domain? */
    // pmacc::AreaMapping<pmacc::type::CORE + pmacc::type::BORDER, MappingDesc> mapper(*mapping);
    pmacc::AreaMapping<pmacc::type::CORE + pmacc::type::BORDER, MappingDesc> mapper(*mapping);

    /* define a device buffer - defines a buffer on each device? */
    using Buffer = pmacc::HostDeviceBuffer<float, DIM2>;
    std::unique_ptr<Buffer> buff1;
    buff1 = std::make_unique<Buffer>(layout.getDataSpace());
    auto deviceDB = buff1->getDeviceBuffer().getDataBox(); // gets data box of the buffer

    /* Make locktep workercfg, takes in supercell size */
    auto workerCfg = pmacc::lockstep::makeWorkerCfg(typename MappingDesc::SuperCellSize{});

    PMACC_LOCKSTEP_KERNEL(HelloKernel{}, workerCfg)
    (mapper.getGridDim())(deviceDB, gc.getGlobalRank(), subGrid.getLocalDomain().offset, mapper);

    buff1->deviceToHost();

    /* Read position as dataspace */
    // pmacc::DataSpace<DIM2> readPos{20u, 10u};
    // db(readPos) = 1;
    // std::cout << db(readPos);

    /* Finalize */
    pmacc::Environment<DIM2>::get().finalize();

    return 0;
}
