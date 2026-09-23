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

#include "picongpu/fields/FieldJ.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>

#include <cstdint>
#include <utility>

#include <caravan/core.hpp>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Functor for the stage of the PIC loop setting the current values to zero
            struct CurrentReset
            {
                /** Start setting all current density values to zero.
                 *
                 * J writers must depend on the returned event; the field and its
                 * buffer must outlive it.
                 *
                 * @param context simulation-owned operation scope
                 * @param previous completion of prior J access
                 * @param step index of time iteration (unused)
                 * @return completion of the device fill
                 */
                [[nodiscard]] caravan::Event operator()(
                    caravan::ControlContext& context,
                    caravan::Event previous,
                    uint32_t const) const
                {
                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto& fieldJ = *dc.get<FieldJ>(FieldJ::getName());
                    FieldJ::ValueType zeroJ(FieldJ::ValueType::create(0._X));
                    return fieldJ.assign(context, zeroJ, std::move(previous));
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
