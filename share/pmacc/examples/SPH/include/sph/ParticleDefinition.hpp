#pragma once

#include "sph/param/memory.param"
#include "spmacc/Frame.hpp"
#include "spmacc/ParticleDescription.hpp"

#include <pmacc/identifier/value_identifier.hpp>
#include <pmacc/meta/Pair.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/memory/dataTypes/ListPointer.hpp>
#include <pmacc/particles/memory/dataTypes/StaticArray.hpp>

namespace sph
{
    namespace detail
    {
        /** create static array
         */
        template<uint32_t T_size>
        struct OperatorCreatePairStaticArray
        {
            template<typename X>
            struct apply
            {
                using type = pmacc::meta::Pair<
                    X,
                    pmacc::StaticArray<
                        typename pmacc::traits::Resolve<X>::type::type,
                        std::integral_constant<uint32_t, T_size>>>;
            };
        };
    } // namespace detail

    value_identifier_func(
        uint64_t,
        particleId,
        /* called when particle is created */
        ([](auto const& worker, pmacc::IdGenerator& idGen) constexpr { return idGen.fetchInc(worker); }),
        /* called when particle is copied */
        (pmacc::particles::identifier::CallCopy{}),
        /* called when particle is derived from other particle */
        (pmacc::particles::identifier::CallInitValue{}));

    using ParticleDesc = pmacc::sph::ParticleDescription<
        PMACC_CSTRING("fluid"),
        std::integral_constant<uint32_t, numFrameSlots>,
        pmacc::MakeSeq_t<particleId, pmacc::multiMask>>;

    /** linked list pointer */
    using LinkedListPointer = pmacc::MakeSeq_t<pmacc::NextFramePtr<>>;

    /* extent particle description with pointer to a frame*/
    using FrameDescription = typename pmacc::sph::ReplaceFrameExtensionSeq<ParticleDesc, LinkedListPointer>::type;

    /** frame definition
     *
     * a group of particles is stored as frame
     */
    using FrameType
        = pmacc::sph::Frame<detail::OperatorCreatePairStaticArray<ParticleDesc::NumSlots::value>, FrameDescription>;

} // namespace sph
