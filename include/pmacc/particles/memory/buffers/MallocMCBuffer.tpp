/* Copyright 2015-2024 Rene Widera, Alexander Grund
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

#if (ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)

#    include "pmacc/alpakaHelper/Device.hpp"
#    include "pmacc/math/Vector.hpp"
#    include "pmacc/particles/memory/buffers/MallocMCBuffer.hpp"
#    include "pmacc/types.hpp"

#    include <memory>

namespace pmacc
{
    template<typename T_DeviceHeap>
    MallocMCBuffer<T_DeviceHeap>::MallocMCBuffer(DeviceHeap& deviceHeap)
        : /* currently mallocMC has only one heap */
        deviceHeapInfo(deviceHeap.getHeapLocations()[0])
        , hostBufferOffset(0)
    {
    }

    template<typename T_DeviceHeap>
    MallocMCBuffer<T_DeviceHeap>::~MallocMCBuffer()
    {
    }

    template<typename T_DeviceHeap>
    template<typename T_Queue>
    auto MallocMCBuffer<T_DeviceHeap>::synchronize(T_Queue& queue)
    {
        auto const extent = pmacc::math::Vector<pmacc::MemIdxType, 1>(deviceHeapInfo.size).toAlpakaMemVec();
        if(!hostBuffer)
        {
            hostBuffer = alpaka::allocMappedBufIfSupported<uint8_t, MemIdxType>(
                manager::Device<HostDevice>::get().current(),
                manager::Device<ComputeDevice>::get().getPlatform(),
                extent);
            hostBufferOffset = static_cast<int64_t>(
                reinterpret_cast<uint8_t*>(deviceHeapInfo.p) - alpaka::getPtrNative(*hostBuffer));
        }
        auto host = *hostBuffer;
        auto device = ::alpaka::ViewPlainPtr<ComputeDevice, uint8_t, AlpakaDim<DIM1>, pmacc::MemIdxType>(
            static_cast<uint8_t*>(deviceHeapInfo.p),
            manager::Device<ComputeDevice>::get().current(),
            extent);
        return caravan::alpaka::submit(
            queue,
            [host = std::move(host), device, extent](T_Queue& nativeQueue) mutable
            { alpaka::memcpy(nativeQueue, host, device, extent); });
    }

} // namespace pmacc

#endif
