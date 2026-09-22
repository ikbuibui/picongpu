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
            /** Named completion milestones of the particle push stage.
             *
             * `pushed` completes when every species has been pushed and its boundary conditions
             * have been applied. `communicated` completes when every species' halo exchange that
             * started from its own push has finished. Keeping them separate lets orchestration
             * overlap unrelated field work with the exchange.
             */
            struct ParticlePushEvents
            {
                //! all species pushed and boundaries applied
                caravan::Event pushed;
                //! all species communicated after their own push
                caravan::Event communicated;
            };

            //! Functor for the stage of the PIC loop performing particle push
            struct ParticlePush
            {
                /** Push and communicate all particle species with a pusher.
                 *
                 * Each species' communication depends on its own push/boundary completion, so a
                 * species exchange is not delayed by another species' push. All pushes depend on
                 * @p predecessor, which must cover the prior-step completion and any pre-push
                 * momentum/particle writers.
                 *
                 * @param context simulation-owned operation scope
                 * @param predecessor completion of prior-step and pre-push dependencies
                 * @param currentStep current time iteration
                 * @return completion milestones for push and communication
                 */
                [[nodiscard]] ParticlePushEvents operator()(
                    caravan::ControlContext& context,
                    caravan::Event predecessor,
                    uint32_t const currentStep) const;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
