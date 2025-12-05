#pragma once

#include "spmacc/Frame.hpp"
#include "spmacc/SinglyLinkedListDevice.hpp"

#include <pmacc/traits/IsSpecializationOf.hpp>

#include <cstdint>

namespace pmacc
{

    namespace sph
    {
        template<concepts::SpecializationOf<Frame> T_Frame, typename T_DeviceHeapHandle>
        struct FrameList
        {
            FrameList(T_DeviceHeapHandle const& deviceHeapHandle) : frameList{deviceHeapHandle}
            {
            }

            //! get number of particle in the last frame
            constexpr uint32_t getSizeLastFrame() const
            {
                constexpr uint32_t frameSize = T_Frame::frameSize;

                /* NOTE on result expression understanding:
                 * (numParticles % frameSize) =^= how many particle did not fit in a full frame?
                 *
                 * but we need how many are in the last frame,
                 * => (numParticles - 1u) % frameSize + 1u
                 *   only shift by one which is reversed by + 1u
                 * => will return the same result for numParticles =/= i * frameSize ;i \in N
                 * and for numParticles == i * frameSize, i \in N it will return
                 *  ((frameSize * i) - 1u) % frameSize + 1u = (frameSize - 1u) + 1u = frameSize
                 */
                // avoids underflow for uint32_t numParticles = 0u
                return numParticles ? ((numParticles - 1u) % frameSize + 1u) : 0u;
            }

            constexpr uint32_t getNumParticles() const
            {
                return numParticles;
            }

            constexpr void setNumParticles(uint32_t const size)
            {
                numParticles = size;
            }

        private:
            pmacc::sph::SingleLinkedListDevice<T_Frame, T_DeviceHeapHandle> frameList;
            uint32_t numParticles = 0u;
        };
    } // namespace sph
} // namespace pmacc
