#include "picongpu/simulation/stage/ParticleInit.hpp"

// clang-format off
#include "picongpu/defines.hpp"
#include "picongpu/particles/param.hpp"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/manipulators/manipulators.hpp"
#include "picongpu/param/particleFilters.param"
#include "picongpu/param/speciesInitialization.param"
// clang-format on

#include "picongpu/particles/boundary/RemoveOuterParticles.hpp"
#include "picongpu/particles/filter/filter.hpp"
#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/particles/traits/FilterByFlag.hpp>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

#include <cstdint>
#include <utility>

namespace picongpu::simulation::stage
{
    namespace particles
    {
        /** Remove all particles of the species that are outside the respective boundaries
         *
         * Must be called only for species with a pusher
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of particle species that is checked
         */
        template<typename T_SpeciesType>
        struct RemoveOuterParticles
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            /** @return lazy sender removing the species' outer particles and filling frame gaps */
            HINLINE auto operator()(uint32_t const currentStep) const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                return picongpu::particles::boundary::removeOuterParticles(*species, currentStep);
            }
        };

        //! Remove all particles of all species with pusher flag that are outside the respective boundaries
        struct RemoveOuterParticlesAllSpecies
        {
            /** @return lazy sender removing outer particles for every species in order */
            template<typename... TSpecies>
            HINLINE auto removeAll(uint32_t const currentStep, pmacc::mp_list<TSpecies...>) const
            {
                return ::picongpu::particles::detail::sequenceAll(RemoveOuterParticles<TSpecies>{}(currentStep)...);
            }

            HINLINE auto operator()(uint32_t const currentStep) const
            {
                using VectorSpeciesWithPusher =
                    typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, particlePusher<>>::type;
                return removeAll(currentStep, VectorSpeciesWithPusher{});
            }
        };

        /** Initialize a species' supercell/frame storage.
         *
         * PMacc allocation does not zero storage; density creation writes into the supercell
         * list heads, so this reset must complete before the initialization pipeline.
         */
        template<typename T_SpeciesType>
        struct ResetSpeciesStorage
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE auto operator()() const
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto species = dc.get<SpeciesType>(FrameType::getName());
                return species->resetAsync();
            }
        };

        /** Initialize supercell storage for every species in order. */
        template<typename... TSpecies>
        HINLINE caravan::Event resetSpeciesStorage(
            caravan::ControlContext& context,
            pmacc::mp_list<TSpecies...>)
        {
            auto& device = Environment<>::get().DeviceContext();
            caravan::Event previous;
            (
                (previous = context.spawn(caravan::alpaka::withDevice(
                     device,
                     caravan::asSender(previous) | caravan::sequence(ResetSpeciesStorage<TSpecies>{}())))),
                ...);
            return previous;
        }

        /** Sequentially run every initialization functor, each depending on its predecessor.
         *
         * Functors are started eagerly through the context; the returned event is the completion of
         * the last functor and therefore of the whole pipeline. This is the single initialization
         * boundary, so a caller may wait on it once.
         *
         * @return completion of the last initialization functor
         */
        template<typename... TFunctors>
        HINLINE caravan::Event runInitPipeline(
            caravan::ControlContext& context,
            caravan::Event previous,
            uint32_t const currentStep,
            pmacc::mp_list<TFunctors...>)
        {
            auto& device = Environment<>::get().DeviceContext();
            ((previous = context.spawn(caravan::alpaka::withDevice(
                  device,
                  caravan::asSender(previous) | caravan::sequence(TFunctors{}(currentStep))))),
             ...);
            return previous;
        }
    } // namespace particles

    caravan::Event ParticleInit::operator()(caravan::ControlContext& context, uint32_t const step) const
    {
        /* Zero supercell/frame storage before anything writes into it. */
        auto previous = particles::resetSpeciesStorage(context, VectorAllSpecies{});
        previous = particles::runInitPipeline(context, std::move(previous), step, picongpu::particles::InitPipeline{});
        /* Remove all particles that are outside the respective boundaries
         * (this can happen if density functor didn't account for it).
         * For the rest of the simulation we can be sure the only external particles just crossed the
         * border.
         */
        auto& device = Environment<>::get().DeviceContext();
        return context.spawn(caravan::alpaka::withDevice(
            device,
            caravan::asSender(std::move(previous))
                | caravan::sequence(particles::RemoveOuterParticlesAllSpecies{}(step))));
    }

} // namespace picongpu::simulation::stage
