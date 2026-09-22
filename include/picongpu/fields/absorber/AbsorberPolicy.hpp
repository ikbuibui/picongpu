/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Sergei Bastrakov
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

#include <array>
#include <cstdint>
#include <stdexcept>

namespace picongpu
{
    namespace fields
    {
        namespace absorber
        {
            /** Supported absorber kinds, same for all absorbing boundaries
             *
             * Exponential - exponential damping absorber.
             * None - all boundaries are periodic, no absorber.
             * Pml - perfectly matched layer absorber.
             *
             * Kept dependency-light so the selection policy below can be unit tested without
             * pulling in species/field definitions.
             */
            enum class AbsorberKind : uint32_t
            {
                Exponential,
                None,
                Pml
            };

            /** Reject values outside the supported absorber-kind enumeration.
             *
             * @param kind absorber kind to validate
             * @throw std::runtime_error for a value outside the enum
             */
            inline void validateAbsorberKind(AbsorberKind const kind)
            {
                switch(kind)
                {
                case AbsorberKind::Exponential:
                case AbsorberKind::None:
                case AbsorberKind::Pml:
                    break;
                default:
                    throw std::runtime_error("Unsupported field absorber type");
                }
            }

            /** Validate a kind actually installed on the factory.
             *
             * Ordinary builds accept all three kinds. Minimal mode supports only None; enforcing
             * this on AbsorberFactory::setKind() prevents callers from bypassing the policy.
             * Note this is distinct from effectiveAbsorberKind(): the requested kind may be Pml
             * even in minimal mode, but the effective kind is then validated here.
             *
             * @param kind effective absorber kind to install
             * @throw std::runtime_error for a kind not supported in this build
             */
            inline void validateFactoryKind(AbsorberKind const kind)
            {
                validateAbsorberKind(kind);
#if defined(PICONGPU_MINIMAL_CARAVAN_THERMAL)
                if(kind != AbsorberKind::None)
                    throw std::runtime_error(
                        "PICONGPU_MINIMAL_CARAVAN_THERMAL supports only the 'none' field absorber");
#endif
            }

            /** Effective absorber kind for a requested kind and boundary periodicity.
             *
             * Ordinary builds override to None when all active dimensions are periodic and
             * otherwise use the requested kind. Minimal mode requires all active dimensions to
             * be periodic and supports only None.
             *
             * Only the first @p activeDimensions entries of @p isPeriodic are inspected; inactive
             * dimensions must not affect the policy (e.g. a 2D run in the x-y plane must ignore z).
             *
             * @param requested user-requested absorber kind
             * @param isPeriodic per-axis boundary periodicity, indexed by axis
             * @param activeDimensions number of active simulation dimensions (simDim)
             * @throw std::runtime_error for an unsupported kind or a non-periodic minimal configuration
             */
            inline AbsorberKind effectiveAbsorberKind(
                AbsorberKind const requested,
                std::array<bool, 3> const& isPeriodic,
                uint32_t const activeDimensions)
            {
                validateAbsorberKind(requested);
                if(activeDimensions > 3u)
                    throw std::runtime_error("Invalid number of active dimensions for absorber policy");
#if defined(PICONGPU_MINIMAL_CARAVAN_THERMAL)
                for(uint32_t axis = 0u; axis < activeDimensions; ++axis)
                    if(!isPeriodic[axis])
                        throw std::runtime_error(
                            "PICONGPU_MINIMAL_CARAVAN_THERMAL requires periodic boundaries in all active "
                            "simulation dimensions");
                return AbsorberKind::None;
#else
                for(uint32_t axis = 0u; axis < activeDimensions; ++axis)
                    if(!isPeriodic[axis])
                        return requested;
                /* All active boundaries periodic: override to None. This keeps compatibility with
                 * pre-existing checkpoints and avoids empty PML fields.
                 */
                return AbsorberKind::None;
#endif
            }

        } // namespace absorber
    } // namespace fields
} // namespace picongpu
