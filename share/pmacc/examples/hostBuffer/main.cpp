/* Copyright 2023 Tapish Narwal
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

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>

#include <fstream>
#include <iostream>
#include <memory>

#define NUM_STEPS 1000
#define NUM_DEVICES_PER_DIM 1
#define THERMAL_DIFFUSIVITY 4 // POSITIVE FACTOR
#define DX 4 // GRID SPACING
#define DT 1 // TIME STEP - STABLE IF DT < (DX * DX) / (4 * THERMAL_DIFFUSIVITY)


auto main(int argc, char** argv) -> int
{
    const auto devices = pmacc::DataSpace<DIM2>::create(NUM_DEVICES_PER_DIM);
    const auto periodic = pmacc::DataSpace<DIM2>::create(1);
    pmacc::Environment<DIM2>::get().initDevices(devices, periodic);
    std::cout << devices;
    /** define a gloabl grid */
    const pmacc::DataSpace<DIM2> gridSize{256u, 256u};

    auto& gc = pmacc::Environment<DIM2>::get().GridController();

    /** device local grid size */
    const pmacc::DataSpace<DIM2> localGridSize{gridSize / devices};

    pmacc::Environment<DIM2>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);

    // /** define host buffers, two because we dont do in place writes */
    auto buff1 = std::make_unique<pmacc::HostBuffer<float, DIM2>>(pmacc::DataSpace<DIM2>::create(1));

    pmacc::Environment<DIM2>::get().finalize();

    return 0;
}
