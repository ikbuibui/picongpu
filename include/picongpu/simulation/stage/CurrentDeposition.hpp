/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Richard Pausch, Alexander Debus, Marco Garten,
 *                     Benjamin Worpitz, Alexander Grund, Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstdint>

#include <caravan/core.hpp>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop performing current deposition
            struct CurrentDeposition
            {
                /** Compute the current created by particles and add it to the current
                 *  density.
                 *
                 * @param context simulation-owned operation scope
                 * @param previous completion of all prerequisites (exchange, field pre-update, reset)
                 * @param step index of time iteration
                 * @return completion of all deposition kernels
                 */
                [[nodiscard]] caravan::Event operator()(
                    caravan::ControlContext& context,
                    caravan::Event previous,
                    uint32_t const step) const;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
