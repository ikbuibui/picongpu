/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <pmacc/Environment.hpp>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

#include <utility>

namespace picongpu::fields::detail
{
    /** Start a field-buffer host-to-device copy after all conflicting producers.
     *
     * Host writes to the buffer are a conflicting access and must precede this call;
     * device consumers must depend on the returned event. The buffer, its host
     * storage, and the device context must outlive completion. Host contents are
     * preserved and copied, but no host access may overlap the transfer.
     */
    template<typename T_Buffer>
    [[nodiscard]] caravan::Event upload(
        caravan::ControlContext& context,
        T_Buffer& buffer,
        caravan::Event previous = {})
    {
        auto& device = pmacc::Environment<>::get().DeviceContext();
        return context.spawn(
            caravan::alpaka::withDevice(
                device,
                caravan::asSender(std::move(previous)) | caravan::sequence(buffer.hostToDevice())));
    }

    /** Start a field-buffer device-to-host copy after all conflicting writers.
     *
     * Device writers must precede this call; host observation or reuse of the
     * buffer is valid only after the returned event completes. The buffer, its
     * host storage, and the device context must outlive completion.
     */
    template<typename T_Buffer>
    [[nodiscard]] caravan::Event download(
        caravan::ControlContext& context,
        T_Buffer& buffer,
        caravan::Event previous = {})
    {
        auto& device = pmacc::Environment<>::get().DeviceContext();
        return context.spawn(
            caravan::alpaka::withDevice(
                device,
                caravan::asSender(std::move(previous)) | caravan::sequence(buffer.deviceToHost())));
    }

    /** Restore full field extent and clear device storage after a predecessor.
     *
     * Host contents are preserved; only device storage is cleared to @p zero for
     * the full restored capacity. Metadata mutation and the clear begin only after
     * the predecessor completes on the control loop. The buffer and device context
     * must outlive completion, and no conflicting host or device access may overlap.
     */
    template<typename T_Buffer, typename T_Value>
    [[nodiscard]] caravan::Event reset(
        caravan::ControlContext& context,
        T_Buffer& buffer,
        T_Value zero,
        caravan::Event previous = {})
    {
        return context.spawn(
            context.onControl(caravan::asSender(std::move(previous)))
            | caravan::letValue([&buffer, zero]
            {
                buffer.getHostBuffer().reset(true);
                auto& deviceBuffer = buffer.getDeviceBuffer();
                deviceBuffer.setSizeHostSide(deviceBuffer.capacityND().productOfComponents());
                auto& device = pmacc::Environment<>::get().DeviceContext();
                return caravan::alpaka::withDevice(device, deviceBuffer.setValue(zero));
            }));
    }
} // namespace picongpu::fields::detail
