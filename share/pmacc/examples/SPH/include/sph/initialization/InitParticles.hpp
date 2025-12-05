#pragma once

namespace sph
{

    namespace detail
    {
        struct InitParticleSystem
        {
            constexpr auto operator()(auto particleFrameList /** init args */) const
            {
                // calculate how many particles we need to make in this system
                // allocate all frames we need, if particles wont fit into the leftover space in the current frame
                // fill frames in parallel
                // add these frames to the list
            }
        };

    } // namespace detail

    /** Kernel which fills the simulation with particles
     * goes over all the ParticleSystems and then initializes them using the densities
     *
     */
    // do we do boundaries here?
    struct InitParticles
    {
        constexpr auto operator()(/** init args */) const
        {
            /** Launched with max blocks and num threads per block we can do.
             * for each particle system we want to assign as many blocks as it needs for its work
             *
             * - calculate number of blocks needed for each system. Based on the density functors etc
             * - do a scan, and create a mapping from frame idx to
             * - we need some natural order of particle initialization, which can be split by number of frame slots so
             * that particle init can be independent across blocks and threads
             *
             *
             */


            // for(auto& particlePatch : ParticleSystemsContainer)
            // {
            //    InitParticleSystem{}(particlePatch.getParticleFrameList());
            // }
        }
    };


} // namespace sph
