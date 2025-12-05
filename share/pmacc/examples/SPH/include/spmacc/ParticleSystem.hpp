#pragma once

#include "spmacc/FrameList.hpp"

#include <pmacc/traits/IsSpecializationOf.hpp>

namespace pmacc
{
    namespace sph
    {
        /**
         * Holds a defined volume and the paricles in that volume
         * Maybe this should be held in an SoA, to do quick ops on the Volume/Frame
         */
        template<typename TVolume, concepts::SpecializationOf<FrameList> TParticleFrameList>
        struct ParticleSystem
        {
            constexpr ParticleSystem(auto const& deviceHeapHandle) : particleFrameList{deviceHeapHandle}
            {
            }

            // return a reference to the Frame list
            auto& getParticleFrameList()
            {
                return particleFrameList;
            }


        private:
            TVolume volume;
            TParticleFrameList particleFrameList;
        };
    } // namespace sph
} // namespace pmacc
