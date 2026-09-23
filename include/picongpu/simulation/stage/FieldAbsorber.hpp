/* Copyright 2021-2024 Sergei Bastrakov
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

#include "picongpu/fields/absorber.hpp"

#include <pmacc/Environment.hpp>

#include <boost/program_options/options_description.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing field absorption
             *
             * This stage does not run by itself, but is needed to propagate command-line parameters
             */
            class FieldAbsorber
            {
            public:
                /** Register program options for field absorber
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
#if !defined(PICONGPU_MINIMAL_CARAVAN_THERMAL)
                    desc.add_options()(
                        "fieldAbsorber",
                        po::value<std::string>(&kindName),
                        std::string(
                            "Field absorber kind [exponential, pml] default: " + kindName
                            + ".\nWhen changing absorber, adjust parameters in fieldAbsorber.param")
                            .c_str());
#else
                    /* Minimal mode is periodic/no-absorber only. Do not advertise an option
                     * that the policy would ignore or reject.
                     */
                    static_cast<void>(desc);
#endif
                }

                /** Load the stage during loading of the simulation.
                 *
                 * This has to be called before any absorber instance or implementation can be safely used.
                 */
                void load()
                {
                    using namespace fields::absorber;
                    AbsorberKind requested = AbsorberKind::None;
                    /* For the all-periodic boundaries case the policy overrides the user's choice
                     * and uses None. This keeps compatibility with pre-existing checkpoints and
                     * avoids empty PML fields. Keep the ordinary parse behavior unchanged.
                     */
                    if(!areAllBoundariesPeriodic())
                    {
                        if(kindName == "exponential")
                            requested = AbsorberKind::Exponential;
                        else if(kindName == "pml")
                            requested = AbsorberKind::Pml;
                        else
                            throw std::runtime_error("Unsupported field absorber type");
                    }
                    DataSpace<DIM3> const isPeriodicBoundary
                        = Environment<simDim>::get().GridController().getCommunicator().getPeriodic();
                    std::array<bool, 3> isPeriodic{};
                    for(uint32_t axis = 0u; axis < 3u; ++axis)
                        isPeriodic[axis] = static_cast<bool>(isPeriodicBoundary[axis]);
                    /* The policy runs here, during stage load and before any absorber instance is
                     * constructed by the field solver, so an unsupported configuration fails early.
                     */
                    auto& absorberFactory = AbsorberFactory::get();
                    absorberFactory.setKind(effectiveAbsorberKind(requested, isPeriodic, simDim));
                }

            private:
                //! Name set by program option
                std::string kindName = "pml";

                //! Return whether all boudaries are periodic
                bool areAllBoundariesPeriodic() const
                {
                    DataSpace<DIM3> const isPeriodicBoundary
                        = Environment<simDim>::get().GridController().getCommunicator().getPeriodic();
                    for(uint32_t axis = 0u; axis < simDim; axis++)
                        if(!isPeriodicBoundary[axis])
                            return false;
                    return true;
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
