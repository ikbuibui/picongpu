/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/types.hpp"

#include <optional>
#include <type_traits>
#include <utility>

#include <caravan/alpaka.hpp>

namespace pmacc
{
    namespace detail
    {
        template<typename T_Value>
        struct IndirectValue
        {
            T_Value const* ptr;
        };

        template<typename T_Value>
        constexpr bool isIndirectValue = false;

        template<typename T_Value>
        constexpr bool isIndirectValue<IndirectValue<T_Value>> = true;

        template<uint32_t T_xChunkSize>
        struct KernelSetValue
        {
            template<typename T_DataBox, typename T_Value, typename T_Size, typename T_Acc, typename T_BlockCfg>
            DINLINE void operator()(
                T_Acc const& acc,
                T_DataBox memBox,
                T_Value const& value,
                T_Size const& size,
                T_BlockCfg const& blockCfg) const
            {
                auto const blockIndex = T_Size(device::getBlockIdx(acc));
                auto blockSize = T_Size::create(1);
                blockSize.x() = T_xChunkSize;
                lockstep::makeForEach<T_xChunkSize>(blockCfg.getWorker(acc))(
                    [&](uint32_t const linearIdx)
                    {
                        auto virtualWorkerIdx = T_Size::create(0);
                        virtualWorkerIdx.x() = linearIdx;
                        auto const idx = blockSize * blockIndex + virtualWorkerIdx;
                        if(idx.x() < size.x())
                        {
                            if constexpr(isIndirectValue<T_Value>)
                                memBox(idx) = *value.ptr;
                            else
                                memBox(idx) = value;
                        }
                    });
            }
        };
    } // namespace detail

    /** N-dimensional device buffer
     *
     * @tparam T_Type datatype of the buffer
     * @tparam T_dim dimension of the buffer
     */
    template<class T_Type, unsigned T_dim>
    class DeviceBuffer : public Buffer<T_Type, T_dim>
    {
        using BufferType = ::alpaka::Buf<ComputeDevice, T_Type, AlpakaDim<DIM1>, MemIdxType>;
        using ViewType = alpaka::ViewPlainPtr<ComputeDevice, T_Type, AlpakaDim<T_dim>, MemIdxType>;
        using CurrentSizeBufferDevice = ::alpaka::Buf<ComputeDevice, size_t, AlpakaDim<DIM1>, MemIdxType>;

        void createSizeOnDeviceBuffers()
        {
            currentSizeBufferDevice.emplace(
                alpaka::allocBuf<size_t, MemIdxType>(
                    manager::Device<ComputeDevice>::get().current(),
                    MemSpace<DIM1>(1).toAlpakaMemVec()));
        }

    public:
        using DataBoxType = typename Buffer<T_Type, T_dim>::DataBoxType;
        std::optional<BufferType> devBuffer;
        std::optional<ViewType> view;
        std::optional<CurrentSizeBufferDevice> currentSizeBufferDevice;

        using BufferType1D = ::alpaka::ViewPlainPtr<ComputeDevice, T_Type, AlpakaDim<DIM1>, MemIdxType>;

        BufferType1D as1DBuffer()
        {
            auto numElements = this->size();
            return BufferType1D(
                alpaka::getPtrNative(*view),
                alpaka::getDev(*devBuffer),
                MemSpace<DIM1>(numElements).toAlpakaMemVec());
        }

        BufferType1D as1DBufferNElem(size_t const numElements)
        {
            PMACC_ASSERT(numElements < this->size());
            return BufferType1D(
                alpaka::getPtrNative(*view),
                alpaka::getDev(*devBuffer),
                MemSpace<DIM1>(numElements).toAlpakaMemVec());
        }

        /** Borrowed view; the DeviceBuffer must outlive all native use. */
        ViewType getAlpakaView() const
        {
            return *view;
        }

        /** View retaining the underlying allocation for asynchronous operation state. */
        auto getOwnedAlpakaView() const
        {
            return caravan::alpaka::OwnedView{*view, *devBuffer};
        }

        /** Lazily fill every current element with a value on the caller-supplied queue. */
        template<typename T_Queue>
        auto setValueAsync(T_Queue& queue, T_Type const& value)
        {
            auto const areaSize = MemSpace<T_dim>(this->sizeND(this->size()));
            auto gridSize = areaSize;
            constexpr uint32_t xChunkSize = 256u;
            gridSize.x() = alpaka::core::divCeil(gridSize.x(), static_cast<size_t>(xChunkSize));
            auto const blockCfg = lockstep::makeBlockCfg<xChunkSize>();
            auto blockSize = DataSpace<T_dim>::create(1);
            blockSize.x() = blockCfg.numWorkers();
            auto const workDiv = alpaka::WorkDivMembers<AlpakaDim<T_dim>, IdxType>{
                gridSize.toAlpakaKernelVec(),
                blockSize.toAlpakaKernelVec(),
                DataSpace<T_dim>::create(1).toAlpakaKernelVec()};
            auto destination = getOwnedAlpakaView();
            auto const destinationBox = getDataBox();

            if constexpr(sizeof(T_Type) <= 128u && std::is_trivially_copyable_v<T_Type>)
                return caravan::alpaka::submit(
                    queue,
                    [destination = std::move(destination), destinationBox, value, areaSize, workDiv, blockCfg](
                        T_Queue& nativeQueue) mutable
                    {
                        if(areaSize.productOfComponents() != 0u)
                            alpaka::exec<Acc<T_dim>>(
                                nativeQueue,
                                workDiv,
                                detail::KernelSetValue<xChunkSize>{},
                                destinationBox,
                                value,
                                areaSize,
                                blockCfg);
                    });
            else
            {
                auto hostValue = alpaka::allocMappedBufIfSupported<T_Type, MemIdxType>(
                    manager::Device<HostDevice>::get().current(),
                    manager::Device<ComputeDevice>::get().getPlatform(),
                    MemSpace<DIM1>(1).toAlpakaMemVec());
                alpaka::getPtrNative(hostValue)[0] = value;
                auto deviceValue = alpaka::allocBuf<T_Type, MemIdxType>(
                    manager::Device<ComputeDevice>::get().current(),
                    MemSpace<DIM1>(1).toAlpakaMemVec());
                return caravan::alpaka::submit(
                    queue,
                    [destination = std::move(destination),
                     destinationBox,
                     hostValue = std::move(hostValue),
                     deviceValue = std::move(deviceValue),
                     areaSize,
                     workDiv,
                     blockCfg](T_Queue& nativeQueue) mutable
                    {
                        if(areaSize.productOfComponents() == 0u)
                            return;
                        alpaka::memcpy(nativeQueue, deviceValue, hostValue, MemSpace<DIM1>(1).toAlpakaMemVec());
                        alpaka::exec<Acc<T_dim>>(
                            nativeQueue,
                            workDiv,
                            detail::KernelSetValue<xChunkSize>{},
                            destinationBox,
                            detail::IndirectValue<T_Type>{alpaka::getPtrNative(deviceValue)},
                            areaSize,
                            blockCfg);
                    });
            }
        }

        /** Allocate uninitialized data accessible from the device.
         *
         * @param size extent for each dimension (in elements)
         * @param sizeOnDevice allocate device-side size storage; its value must be initialized through an explicit
         *                      queue operation before device use
         *
         * @attention offset + size must be less or equal to the size of the source buffer
         */
        DeviceBuffer(MemSpace<T_dim> const& size, bool sizeOnDevice = false)
            : Buffer<T_Type, T_dim>(size)
            , devBuffer(
                  alpaka::allocBuf<T_Type, MemIdxType>(
                      manager::Device<ComputeDevice>::get().current(),
                      MemSpace<DIM1>(size.productOfComponents()).toAlpakaMemVec()))
        {
            MemSpace<T_dim> pitchInBytes;
            pitchInBytes.x() = sizeof(T_Type);
            for(uint32_t d = 1u; d < T_dim; ++d)
                pitchInBytes[d] = pitchInBytes[d - 1u] * size[d - 1u];
            view.emplace(ViewType(
                alpaka::getPtrNative(*devBuffer),
                alpaka::getDev(*devBuffer),
                size.toAlpakaMemVec(),
                pitchInBytes.toAlpakaMemVec()));

            if(sizeOnDevice)
                createSizeOnDeviceBuffers();
            this->isMemoryContiguous = true;
        }

        /** create a shallow view into an existing buffer
         *
         * @param source buffer to create the view on
         * @param size extent for each dimension (in elements)
         * @param offset offset within the source (in elements)
         * @param sizeOnDevice allocate device-side size storage; its value must be initialized through an explicit
         *                      queue operation before device use
         *
         * @attention offset + size must be less or equal to the size of the source buffer
         */
        DeviceBuffer(
            DeviceBuffer<T_Type, T_dim>& source,
            MemSpace<T_dim> size,
            MemSpace<T_dim> offset,
            bool sizeOnDevice = false)
            : Buffer<T_Type, T_dim>(size)
            , devBuffer(source.devBuffer)
        {
            auto subView = createSubView(*source.view, size.toAlpakaMemVec(), offset.toAlpakaMemVec());
            view.emplace(ViewType(
                alpaka::getPtrNative(subView),
                alpaka::getDev(subView),
                alpaka::getExtents(subView),
                alpaka::getPitchesInBytes(subView)));
            if(sizeOnDevice)
                createSizeOnDeviceBuffers();
            this->isMemoryContiguous = T_dim == DIM1;
        }

        ~DeviceBuffer() override = default;

        T_Type* data() override
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            return alpaka::getPtrNative(*view);
        }

        DataBoxType getDataBox() override
        {
            auto pitchBytes = MemSpace<T_dim>(getPitchesInBytes(*view));
            return DataBoxType(PitchedBox<T_Type, T_dim>(alpaka::getPtrNative(*view), pitchBytes));
        }

        /** Show if current size is stored on device.
         *
         * @return return false if no size is stored on device, true otherwise
         */
        bool hasCurrentSizeOnDevice() const
        {
            return currentSizeBufferDevice.has_value();
        }

        /** get the device alpaka buffer with the current size
         *
         * @return device side current size buffer
         */
        CurrentSizeBufferDevice sizeOnDeviceBuffer()
        {
            if(!hasCurrentSizeOnDevice())
            {
                throw std::runtime_error("Buffer has no size on device!, currentSize is only stored on host side.");
            }
            return currentSizeBufferDevice.value();
        }

        /** get the host alpaka buffer with the current size
         *
         * @return host side current size buffer
         */
        typename Buffer<T_Type, T_dim>::CurrentSizeBufferHost sizeHostSideBuffer()
        {
            return this->currentSizeBufferHost;
        }

        /** Host-side cached size. Synchronize sizeOnDeviceBuffer() explicitly when required. */
        size_t size() override
        {
            return Buffer<T_Type, T_dim>::size();
        }

        auto sizeDeviceSideBuffer()
        {
            return currentSizeBufferDevice.value();
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCurrentSize() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            size_t const size = this->size();
            return {alpaka::getPtrNative(*view), size};
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCapacity() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            size_t const size = this->capacityND().productOfComponents();
            return {alpaka::getPtrNative(*view), size};
        }
    };

} // namespace pmacc
