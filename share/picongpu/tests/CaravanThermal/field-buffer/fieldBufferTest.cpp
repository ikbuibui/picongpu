/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 *
 * Focused test for the field-buffer storage operations used by EMFieldBase.
 * It exercises the production helper directly with a real three-component
 * GridBuffer instead of reconstructing the solver/species-dependent field.
 */
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>

#include <cstddef>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/fields/detail/FieldBufferOperations.hpp>

namespace
{
    using Value = pmacc::math::Vector<float, TEST_DIM>;
    using Buffer = pmacc::GridBuffer<Value, TEST_DIM>;

    constexpr Value zeroValue()
    {
        return Value{0.0f, 0.0f, 0.0f};
    }

    Value testValue(float const base)
    {
        Value value;
        for(uint32_t d = 0u; d < TEST_DIM; ++d)
            value[d] = base + static_cast<float>(d);
        return value;
    }

    std::size_t capacityOf(Buffer const& buffer)
    {
        return buffer.getHostBuffer().capacityND().productOfComponents();
    }

    void fillHost(Buffer& buffer, float const base)
    {
        auto* const data = buffer.getHostBuffer().data();
        for(std::size_t i = 0u; i < capacityOf(buffer); ++i)
            data[i] = testValue(base + static_cast<float>(i));
    }

    void uploadImmediately(Buffer& buffer)
    {
        auto& device = pmacc::Environment<>::get().DeviceContext();
        caravan::ControlContext context;
        context.wait(context.spawn(caravan::alpaka::withDevice(device, buffer.hostToDevice())));
    }

    /** Resolve a pending test gate on scope exit so a failing/fatal assertion
     * cannot leave the owned ControlContext joining work that never completes.
     */
    struct GateGuard
    {
        explicit GateGuard(caravan::EventSource& source) : gate(source)
        {
        }

        ~GateGuard()
        {
            gate.setReady();
        }

        caravan::EventSource& gate;
    };
} // namespace

TEST_CASE("Field buffer reset is ordered, restores capacity, and clears device storage", "[picongpu][fields]")
{
    Buffer buffer{pmacc::DataSpace<TEST_DIM>::create(2)};
    auto const capacity = capacityOf(buffer);

    fillHost(buffer, 0.0f);
    uploadImmediately(buffer);

    // Shorten both size metadata values so the reset has to restore them.
    buffer.getHostBuffer().setSizeHostSide(1u);
    buffer.getDeviceBuffer().setSizeHostSide(1u);

    caravan::ControlContext context;
    caravan::EventSource gate;
    GateGuard guard{gate};
    auto const resetEvent = picongpu::fields::detail::reset(context, buffer, zeroValue(), gate.event());
    // Scheduled directly after the reset: the download must observe the restored size.
    auto const downloadEvent = picongpu::fields::detail::download(context, buffer, resetEvent);

    context.runReady();

    // Nothing may mutate while the predecessor is still pending.
    CHECK(buffer.getHostBuffer().size() == 1u);
    CHECK(buffer.getDeviceBuffer().size() == 1u);
    CHECK_FALSE(resetEvent.isReady());
    CHECK_FALSE(downloadEvent.isReady());

    // An independent probe confirms device storage is still the uploaded data.
    context.wait(picongpu::fields::detail::download(context, buffer));
    CHECK(buffer.getHostBuffer().data()[0] == testValue(0.0f));

    gate.setReady();
    context.wait(downloadEvent);

    CHECK(buffer.getHostBuffer().size() == capacity);
    CHECK(buffer.getDeviceBuffer().size() == capacity);
    for(std::size_t i = 0u; i < capacity; ++i)
        CHECK(buffer.getHostBuffer().data()[i] == zeroValue());
}

TEST_CASE("Field buffer reset waits for its predecessor before touching device data", "[picongpu][fields]")
{
    Buffer buffer{pmacc::DataSpace<TEST_DIM>::create(2)};
    auto const capacity = capacityOf(buffer);

    fillHost(buffer, 5.0f);
    uploadImmediately(buffer);

    caravan::ControlContext context;
    caravan::EventSource gate;
    GateGuard guard{gate};
    auto const resetEvent = picongpu::fields::detail::reset(context, buffer, zeroValue(), gate.event());

    context.runReady();
    CHECK_FALSE(resetEvent.isReady());

    // Independent download: the unresolved reset must not have cleared the device.
    context.wait(picongpu::fields::detail::download(context, buffer));
    for(std::size_t i = 0u; i < capacity; ++i)
        CHECK(buffer.getHostBuffer().data()[i] == testValue(5.0f + static_cast<float>(i)));

    gate.setReady();
    context.wait(resetEvent);
}

TEST_CASE("Field buffer upload and download round-trip nonzero values", "[picongpu][fields]")
{
    Buffer buffer{pmacc::DataSpace<TEST_DIM>::create(2)};
    auto const capacity = capacityOf(buffer);

    fillHost(buffer, 10.0f);

    caravan::ControlContext context;
    context.wait(picongpu::fields::detail::upload(context, buffer));

    // Overwrite the host data so the download result is unambiguous.
    auto* const hostData = buffer.getHostBuffer().data();
    for(std::size_t i = 0u; i < capacity; ++i)
        hostData[i] = zeroValue();

    context.wait(picongpu::fields::detail::download(context, buffer));

    for(std::size_t i = 0u; i < capacity; ++i)
        CHECK(hostData[i] == testValue(10.0f + static_cast<float>(i)));
}

// NOTE: A failed-predecessor non-mutation case is not representable at this
// revision. `caravan::CompletionState` only distinguishes pending from ready;
// submission and connection failures are fatal by contract, so there is no
// failed Event that can be installed as a `previous` predecessor.
