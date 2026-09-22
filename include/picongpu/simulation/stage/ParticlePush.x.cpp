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

#include "picongpu/simulation/stage/ParticlePush.hpp"

#include "picongpu/defines.hpp"
#include "picongpu/particles/boundary/Apply.hpp"
#include "picongpu/particles/param.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/Communication.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace picongpu
{
    namespace particles
    {
        /** Push a species and apply its boundary conditions
         *
         * Both operations only affect species with a pusher.
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct PushSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            /** @return completion of the species push and boundary application */
            HINLINE caravan::Event operator()(
                caravan::ControlContext& context,
                caravan::Event predecessor,
                uint32_t const currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                auto pushed = species->update(context, std::move(predecessor), currentStep);
                return species->applyBoundary(context, std::move(pushed), currentStep);
            }
        };

        /** Communicate a species after its own push completion
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct CommunicateSpecies
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            /** @return completion of the species halo exchange */
            HINLINE caravan::Event operator()(caravan::ControlContext& context, caravan::Event pushed) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                return pmacc::particles::spawnCommunication(context, *species, std::move(pushed));
            }
        };
    } // namespace particles

    namespace simulation
    {
        namespace stage
        {
            namespace detail
            {
                /** Push every species and start its own communication.
                 *
                 * The two arrays preserve per-species dependency: `communicated[i]` depends on
                 * `pushed[i]`, not on the joined push milestone.
                 */
                template<typename... TSpecies, std::size_t... TIndex>
                HINLINE ParticlePushEvents pushAndCommunicate(
                    caravan::ControlContext& context,
                    caravan::Event predecessor,
                    uint32_t const currentStep,
                    pmacc::mp_list<TSpecies...>,
                    std::index_sequence<TIndex...>)
                {
                    std::array<caravan::Event, sizeof...(TSpecies)> pushed{
                        particles::PushSpecies<TSpecies>{}(context, predecessor, currentStep)...};
                    std::array<caravan::Event, sizeof...(TSpecies)> communicated{
                        particles::CommunicateSpecies<TSpecies>{}(context, pushed[TIndex])...};
                    return {
                        caravan::whenAll(std::span<caravan::Event const>{pushed}),
                        caravan::whenAll(std::span<caravan::Event const>{communicated})};
                }
            } // namespace detail

            ParticlePushEvents ParticlePush::operator()(
                caravan::ControlContext& context,
                caravan::Event predecessor,
                uint32_t const currentStep) const
            {
                using VectorSpeciesWithPusher =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
                constexpr auto numSpecies = pmacc::mp_size<VectorSpeciesWithPusher>::value;
                return detail::pushAndCommunicate(
                    context,
                    std::move(predecessor),
                    currentStep,
                    VectorSpeciesWithPusher{},
                    std::make_index_sequence<numSpecies>{});
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
