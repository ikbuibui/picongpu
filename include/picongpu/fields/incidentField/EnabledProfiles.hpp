/* Copyright 2020-2024 Sergei Bastrakov
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

#include "picongpu/defines.hpp"
#include "picongpu/fields/incidentField/param.hpp"

#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/meta/conversion/Unique.hpp>

#include <boost/mp11/algorithm.hpp>

#include <cstdint>
#include <type_traits>

namespace picongpu::fields::incidentField
{
    //! Typelist of all enabled profiles, can contain duplicates
    using EnabledProfiles = pmacc::MakeSeq_t<
        XMin,
        XMax,
        YMin,
        YMax,
        std::conditional_t<simDim == 3, pmacc::MakeSeq_t<ZMin, ZMax>, pmacc::MakeSeq_t<>>>;

    //! Typelist of all unique enabled profiles, can contain duplicates
    using UniqueEnabledProfiles = pmacc::Unique_t<EnabledProfiles>;

// The profile declaration only exists when openPMD is enabled, so the detection
// trait and its assertion must be guarded by both conditions.
#if defined(PICONGPU_MINIMAL_CARAVAN_THERMAL) && (ENABLE_OPENPMD == 1)
    namespace detail
    {
        /** Detect the from-openPMD pulse by partial specialization.
         *
         * Incident-field dispatch goes through trait specializations, so a class-body
         * assertion in the profile itself would not necessarily fire on selection.
         */
        template<typename T_Profile>
        struct IsFromOpenPMDPulse : std::false_type
        {
        };

        template<typename T_Params>
        struct IsFromOpenPMDPulse<profiles::FromOpenPMDPulse<T_Params>> : std::true_type
        {
        };
    } // namespace detail

    static_assert(
        !boost::mp11::mp_any_of<EnabledProfiles, detail::IsFromOpenPMDPulse>::value,
        "PICONGPU_MINIMAL_CARAVAN_THERMAL does not support the FromOpenPMDPulse incident-field profile");
#endif

#if defined(PICONGPU_MINIMAL_CARAVAN_THERMAL)
    namespace detail
    {
        /** Detect the disabled incident-field profile. */
        template<typename T_Profile>
        struct IsNoneProfile : std::false_type
        {
        };

        template<>
        struct IsNoneProfile<profiles::None> : std::true_type
        {
        };
    } // namespace detail

    /* Non-None incident-field profiles update fields through discarded lazy kernels
     * (`updateField`), so the slice rejects them explicitly rather than silently
     * skipping the incident-field contribution.
     */
    static_assert(
        boost::mp11::mp_all_of<EnabledProfiles, detail::IsNoneProfile>::value,
        "PICONGPU_MINIMAL_CARAVAN_THERMAL supports only the None incident-field profile");
#endif
} // namespace picongpu::fields::incidentField
