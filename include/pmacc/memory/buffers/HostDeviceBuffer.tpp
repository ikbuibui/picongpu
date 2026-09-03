/* Copyright 2016-2024 Alexander Grund
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

#include "HostDeviceBuffer.hpp"

#include <type_traits>
#include <utility>

#include <caravan/alpaka.hpp>

namespace pmacc
{
    namespace detail
    {
        template<uint32_t T_dim>
        MemSpace<T_dim> copyExtent(size_t size, MemSpace<T_dim> const& capacity)
        {
            MemSpace<T_dim> extent;
            if constexpr(T_dim == DIM1)
                extent.x() = size;
            else if constexpr(T_dim == DIM2)
            {
                extent.x() = size <= capacity.x() ? size : capacity.x();
                extent.y() = size <= capacity.x() ? 1u : (size + capacity.x() - 1u) / capacity.x();
            }
            else
            {
                extent.x() = size <= capacity.x() ? size : capacity.x();
                extent.y() = size <= capacity.x() ? 1u : capacity.y();
                extent.z() = size <= capacity.x() * capacity.y()
                                 ? 1u
                                 : (size + capacity.x() * capacity.y() - 1u) / (capacity.x() * capacity.y());
                if(size > capacity.x() && size <= capacity.x() * capacity.y())
                    extent.y() = (size + capacity.x() - 1u) / capacity.x();
            }
            return extent;
        }

        template<uint32_t T_dim, typename T_Queue, typename T_Destination, typename T_Source, typename T_UpdateSize>
        auto copyBuffer(
            T_Queue& queue,
            T_Destination& destinationBuffer,
            T_Source& sourceBuffer,
            T_UpdateSize updateSize)
        {
            auto destination = destinationBuffer.getOwnedAlpakaView();
            auto source = sourceBuffer.getOwnedAlpakaView();
            auto destinationSize = destinationBuffer.getOwnedSizeHostBuffer();
            auto sourceSize = sourceBuffer.getOwnedSizeHostBuffer();
            auto const capacity = sourceBuffer.capacityND();
            bool const contiguous = destinationBuffer.isContiguous() && sourceBuffer.isContiguous();
            return caravan::alpaka::submit(
                queue,
                [destination = std::move(destination),
                 source = std::move(source),
                 destinationSize = std::move(destinationSize),
                 sourceSize = std::move(sourceSize),
                 capacity,
                 contiguous,
                 updateSize = std::move(updateSize)](T_Queue& nativeQueue) mutable
                {
                    auto const size = *::alpaka::getPtrNative(sourceSize);
                    *::alpaka::getPtrNative(destinationSize) = size;
                    updateSize(nativeQueue, destinationSize);
                    if(contiguous)
                    {
                        using DestinationView = std::remove_cvref_t<decltype(destination.view)>;
                        using SourceView = std::remove_cvref_t<decltype(source.view)>;
                        using DestinationFlatView = ::alpaka::ViewPlainPtr<
                            ::alpaka::Dev<DestinationView>,
                            ::alpaka::Elem<DestinationView>,
                            AlpakaDim<DIM1>,
                            MemIdxType>;
                        using SourceFlatView = ::alpaka::ViewPlainPtr<
                            ::alpaka::Dev<SourceView>,
                            ::alpaka::Elem<SourceView>,
                            AlpakaDim<DIM1>,
                            MemIdxType>;
                        auto const extent = MemSpace<DIM1>{size}.toAlpakaMemVec();
                        DestinationFlatView destinationView(
                            ::alpaka::getPtrNative(destination.view),
                            ::alpaka::getDev(destination.view),
                            extent);
                        SourceFlatView sourceView(
                            ::alpaka::getPtrNative(source.view),
                            ::alpaka::getDev(source.view),
                            extent);
                        ::alpaka::memcpy(nativeQueue, destinationView, sourceView, extent);
                    }
                    else
                    {
                        auto const extent = copyExtent<T_dim>(size, capacity).toAlpakaMemVec();
                        ::alpaka::memcpy(nativeQueue, destination.view, source.view, extent);
                    }
                });
        }
    } // namespace detail

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(DataSpace<T_dim> const& size, bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBuffer<T_Type, T_dim>>(size);
        deviceBuffer = std::make_unique<DeviceBuffer<T_Type, T_dim>>(size, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        DBuffer& otherDeviceBuffer,
        DataSpace<T_dim> const& size,
        bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBuffer<T_Type, T_dim>>(size);
        deviceBuffer = std::make_unique<DeviceBufferType>(otherDeviceBuffer, size, DataSpace<T_dim>(), sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostDeviceBuffer<T_Type, T_dim>::HostDeviceBuffer(
        HBuffer& otherHostBuffer,
        DataSpace<T_dim> const& offsetHost,
        DBuffer& otherDeviceBuffer,
        DataSpace<T_dim> const& offsetDevice,
        GridLayout<T_dim> const size,
        bool sizeOnDevice)
    {
        hostBuffer = std::make_unique<HostBufferType>(otherHostBuffer, size, offsetHost);
        deviceBuffer = std::make_unique<DeviceBufferType>(otherDeviceBuffer, size, offsetDevice, sizeOnDevice);
    }

    template<typename T_Type, unsigned T_dim>
    HostBuffer<T_Type, T_dim>& HostDeviceBuffer<T_Type, T_dim>::getHostBuffer() const
    {
        return *hostBuffer;
    }

    template<typename T_Type, unsigned T_dim>
    DeviceBuffer<T_Type, T_dim>& HostDeviceBuffer<T_Type, T_dim>::getDeviceBuffer() const
    {
        return *deviceBuffer;
    }

    template<typename T_Type, unsigned T_dim>
    void HostDeviceBuffer<T_Type, T_dim>::reset(bool preserveData)
    {
        deviceBuffer->reset(preserveData);
        hostBuffer->reset(preserveData);
    }

    template<typename T_Type, unsigned T_dim>
    template<typename T_Queue>
    auto HostDeviceBuffer<T_Type, T_dim>::hostToDevice(T_Queue& queue)
    {
        auto deviceSize = deviceBuffer->currentSizeBufferDevice;
        return detail::copyBuffer<T_dim>(
            queue,
            *deviceBuffer,
            *hostBuffer,
            [deviceSize = std::move(deviceSize)](T_Queue& nativeQueue, auto const& hostSize) mutable
            {
                if(deviceSize)
                    ::alpaka::memcpy(
                        nativeQueue,
                        *deviceSize,
                        hostSize,
                        ::alpaka::Vec<AlpakaDim<DIM1>, MemIdxType>::ones());
            });
    }

    template<typename T_Type, unsigned T_dim>
    void HostDeviceBuffer<T_Type, T_dim>::hostToDevice()
    {
        deviceBuffer->copyFrom(*hostBuffer);
    }

    template<typename T_Type, unsigned T_dim>
    template<typename T_Queue>
    auto HostDeviceBuffer<T_Type, T_dim>::deviceToHost(T_Queue& queue)
    {
        return detail::copyBuffer<T_dim>(queue, *hostBuffer, *deviceBuffer, [](T_Queue&, auto const&) {});
    }

    template<typename T_Type, unsigned T_dim>
    void HostDeviceBuffer<T_Type, T_dim>::deviceToHost()
    {
        hostBuffer->copyFrom(*deviceBuffer);
    }

} // namespace pmacc
