/* Copyright 2024-2024 Rene Widera
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
            //! Initialize particles
            struct ParticleInit
            {
                /** Initialize particles dependent of the given step and return completion.
                 *
                 * The returned event completes when the whole initialization pipeline and the
                 * following removal of outer particles have finished. Callers must wait for it
                 * before reading particles or reusing their buffers.
                 *
                 * @param context simulation-owned operation scope
                 * @param step index of time iteration
                 * @return completion of all particle initialization work
                 */
                [[nodiscard]] caravan::Event operator()(caravan::ControlContext& context, uint32_t const step) const;
            };
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
