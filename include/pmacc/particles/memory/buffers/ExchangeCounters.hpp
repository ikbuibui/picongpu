/* Copyright 2026 PIConGPU contributors
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/types.hpp"

#include <cstdint>
#include <type_traits>

namespace pmacc
{
    /** Element counters written by device kernels and read back on the host.
     *
     * All counters of one group share a single device allocation and a single host allocation.
     * The value moves between the two copies: a reset (TaskResetExchangeCounters) hands device
     * authority to kernels, a publish (TaskPublishExchangeCounters) copies the device result into
     * the host snapshot.
     *
     * Device side counters are plain memory, so writes must be atomic whenever several threads
     * update the same slot; host mapped memory would serialize those atomics.
     *
     * @tparam T_Count counter scalar; must match the atomic type used by the writing kernel
     * @tparam T_numCounters number of counters in the group
     */
    template<typename T_Count, uint32_t T_numCounters>
    class ExchangeCounters
    {
    public:
        using CountType = T_Count;
        using DeviceCounters = ::alpaka::Buf<ComputeDevice, T_Count, AlpakaDim<DIM1>, MemIdxType>;
        using HostCounters = decltype(alpaka::allocMappedBufIfSupported<T_Count, MemIdxType>(
            manager::Device<HostDevice>::get().current(),
            manager::Device<ComputeDevice>::get().getPlatform(),
            MemSpace<DIM1>(T_numCounters).toAlpakaMemVec()));

        static constexpr uint32_t numCounters = T_numCounters;
        static_assert(numCounters > 0u, "an exchange counter group needs at least one counter");

        ExchangeCounters()
            : deviceCounters(
                  alpaka::allocBuf<T_Count, MemIdxType>(
                      manager::Device<ComputeDevice>::get().current(),
                      MemSpace<DIM1>(numCounters).toAlpakaMemVec()))
            , hostCounters(
                  alpaka::allocMappedBufIfSupported<T_Count, MemIdxType>(
                      manager::Device<HostDevice>::get().current(),
                      manager::Device<ComputeDevice>::get().getPlatform(),
                      MemSpace<DIM1>(numCounters).toAlpakaMemVec()))
        {
        }

        /** Device counter storage for kernels; authoritative between reset and publish. */
        CountType* deviceData()
        {
            return alpaka::getPtrNative(deviceCounters);
        }

        /** Published counter value; authoritative after the publish task finished. */
        CountType hostValue(uint32_t const index) const
        {
            PMACC_ASSERT(index < numCounters);
            return alpaka::getPtrNative(hostCounters)[index];
        }

        /** Device allocation, e.g. for the reset fill. */
        DeviceCounters& deviceBuffer()
        {
            return deviceCounters;
        }

        /** Host allocation, e.g. as the destination of the publish copy. */
        HostCounters& hostBuffer()
        {
            return hostCounters;
        }

    private:
        DeviceCounters deviceCounters;
        HostCounters hostCounters;
    };
} // namespace pmacc
