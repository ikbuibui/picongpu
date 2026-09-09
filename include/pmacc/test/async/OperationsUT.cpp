/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/alpakaHelper/acc.hpp>
#include <pmacc/device/Reduce.hpp>
#include <pmacc/fields/Communication.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/particles/memory/buffers/StackExchangeBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <memory>
#include <thread>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    struct MockField
    {
        using SuperCellSize = typename pmacc::math::CT::shrinkTo<pmacc::math::CT::Int<1, 1, 1>, TEST_DIM>::type;
        using MappingDesc = pmacc::MappingDescription<TEST_DIM, SuperCellSize>;
        static constexpr uint32_t dim = TEST_DIM;

        pmacc::GridBuffer<int, TEST_DIM>& getGridBuffer()
        {
            return buffer;
        }

        pmacc::GridBuffer<int, TEST_DIM> buffer{pmacc::DataSpace<TEST_DIM>::create(1)};
    };

    struct Increment
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC void operator()(T_Acc const&, T_View values) const
        {
            ++values[0];
        }
    };
} // namespace

namespace
{
    [[maybe_unused]] auto compileFieldSend(pmacc::ComputeDeviceQueue& queue, MockField& field)
    {
        auto sender = pmacc::fields::sendExchange(queue, field, 1u);
        static_assert(caravan::Sender<decltype(sender)>);
        return sender;
    }

    [[maybe_unused]] auto compileFieldReceive(pmacc::ComputeDeviceQueue& queue, MockField& field)
    {
        auto sender = pmacc::fields::receiveExchange(queue, field, 1u);
        static_assert(caravan::Sender<decltype(sender)>);
        return sender;
    }

    [[maybe_unused]] auto compileParticleStackSizes(
        pmacc::ComputeDeviceQueue& queue,
        pmacc::StackExchangeBuffer<int, int, DIM1>& stack)
    {
        return caravan::alpaka::sequence(stack.resetAsync(queue), stack.publishDeviceSizes(queue));
    }
} // namespace

TEST_CASE("PMacc explicitly composes and owns a local accelerator step", "[async][memory]")
{
    auto const deviceManager = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
    pmacc::ComputeDeviceQueue queue(deviceManager);
    auto const one = pmacc::MemSpace<DIM1>::create(1);
    auto const extent = one.toAlpakaMemVec();
    auto const workExtent = ::alpaka::Vec<pmacc::AlpakaDim<DIM1>, pmacc::IdxType>::ones();
    auto const workDiv
        = ::alpaka::WorkDivMembers<pmacc::AlpakaDim<DIM1>, pmacc::IdxType>{workExtent, workExtent, workExtent};

    auto input = std::make_unique<pmacc::HostBuffer<int, DIM1>>(one);
    auto device = std::make_unique<pmacc::DeviceBuffer<int, DIM1>>(one, true);
    pmacc::HostBuffer<int, DIM1> output(one);
    input->data()[0] = 41;

    auto step = caravan::alpaka::sequence(
        caravan::alpaka::sequence(
            caravan::alpaka::sequence(
                caravan::alpaka::fill(queue, device->getOwnedAlpakaView(), 0u),
                caravan::alpaka::copy(queue, device->getOwnedAlpakaView(), input->getOwnedAlpakaView(), extent)),
            caravan::alpaka::kernel<pmacc::Acc<DIM1>>(
                queue,
                workDiv,
                Increment{},
                caravan::alpaka::retain(device->data(), device->getOwnedAlpakaView()))),
        caravan::alpaka::sequence(
            caravan::alpaka::size(queue, device->sizeOnDeviceBuffer(), device->sizeHostSideBuffer()),
            caravan::alpaka::copy(queue, output.getOwnedAlpakaView(), device->getOwnedAlpakaView(), extent)));

    caravan::ControlContext context;
    auto const applicationThread = std::this_thread::get_id();
    bool continued = false;
    auto completion = context.spawn(
        caravan::then(
            context.onControl(std::move(step)),
            [&]
            {
                CHECK(std::this_thread::get_id() == applicationThread);
                continued = true;
            }));
    input.reset();
    device.reset();
    context.wait(completion);

    CHECK(output.data()[0] == 42);
    CHECK(continued);
}

TEST_CASE("DeviceBuffer value fill is a lazy sender", "[async][memory]")
{
    struct LargeValue
    {
        int values[40];
    };

    auto const device = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
    pmacc::ComputeDeviceQueue queue(device);
    auto const extent = pmacc::MemSpace<DIM1>{3u};

    pmacc::HostDeviceBuffer<int, DIM1> small(extent);
    auto smallFill = small.getDeviceBuffer().setValueAsync(queue, 42);
    static_assert(caravan::Sender<decltype(smallFill)>);
    caravan::syncWait(std::move(smallFill));
    caravan::syncWait(small.deviceToHost(queue));
    for(size_t i = 0u; i < 3u; ++i)
        CHECK(small.getHostBuffer().data()[i] == 42);

    pmacc::HostDeviceBuffer<LargeValue, DIM1> large(extent);
    LargeValue value{};
    value.values[0] = 17;
    value.values[39] = 23;
    caravan::syncWait(large.getDeviceBuffer().setValueAsync(queue, value));
    caravan::syncWait(large.deviceToHost(queue));
    for(size_t i = 0u; i < 3u; ++i)
    {
        CHECK(large.getHostBuffer().data()[i].values[0] == 17);
        CHECK(large.getHostBuffer().data()[i].values[39] == 23);
    }
}

TEST_CASE("Device reduction returns a lazy sender", "[async][reduce]")
{
    auto const device = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
    pmacc::ComputeDeviceQueue queue(device);
    pmacc::HostDeviceBuffer<int, DIM1> input(pmacc::MemSpace<DIM1>{4u});
    input.getHostBuffer().data()[0] = 1;
    input.getHostBuffer().data()[1] = 2;
    input.getHostBuffer().data()[2] = 3;
    input.getHostBuffer().data()[3] = 4;
    caravan::syncWait(input.hostToDevice(queue));

    pmacc::device::Reduce reduce(1024u);
    auto reduction = reduce.reduce(queue, pmacc::math::operation::Add{}, input.getDeviceBuffer().getDataBox(), 4u);
    static_assert(caravan::Sender<decltype(reduction)>);
    auto result = caravan::syncWait<int>(std::move(reduction));

    CHECK(result == 10);
}

TEST_CASE("Host-device buffer queue overloads return lazy copies", "[async][memory]")
{
    auto const device = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
    pmacc::ComputeDeviceQueue queue(device);
    pmacc::HostDeviceBuffer<int, DIM1> buffer(pmacc::MemSpace<DIM1>{2u}, true);
    buffer.getHostBuffer().data()[0] = 41;
    buffer.getHostBuffer().data()[1] = 99;
    buffer.getHostBuffer().setSizeHostSide(1u);

    caravan::ControlContext context;
    context.wait(context.spawn(buffer.hostToDevice(queue)));

    buffer.getDeviceBuffer().setSizeHostSide(0u);
    context.wait(context.spawn(
        caravan::alpaka::size(
            queue,
            buffer.getDeviceBuffer().sizeHostSideBuffer(),
            buffer.getDeviceBuffer().sizeOnDeviceBuffer())));
    CHECK(buffer.getDeviceBuffer().size() == 1u);

    buffer.getHostBuffer().data()[0] = 0;
    buffer.getHostBuffer().data()[1] = 77;
    context.wait(context.spawn(buffer.deviceToHost(queue)));
    CHECK(buffer.getHostBuffer().size() == 1u);
    CHECK(buffer.getHostBuffer().data()[0] == 41);
    CHECK(buffer.getHostBuffer().data()[1] == 77);
}
