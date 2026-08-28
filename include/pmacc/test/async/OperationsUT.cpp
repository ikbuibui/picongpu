/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/Environment.hpp>
#include <pmacc/alpakaHelper/acc.hpp>
#include <pmacc/async.hpp>
#include <pmacc/eventSystem/queues/QueueController.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>

#include <alpaka/alpaka.hpp>

#include <memory>

#include <caravan/alpaka.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    struct Increment
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC void operator()(T_Acc const&, T_View values) const
        {
            ++values[0];
        }
    };
} // namespace

TEST_CASE("PMacc explicitly composes and owns a local accelerator step", "[async][memory]")
{
    auto& queue = pmacc::Environment<>::get().QueueController().getNextStream()->borrowAlpakaQueue();
    auto const one = pmacc::MemSpace<DIM1>::create(1);
    auto const extent = one.toAlpakaMemVec();
    auto const workExtent = ::alpaka::Vec<pmacc::AlpakaDim<DIM1>, pmacc::IdxType>::ones();
    auto const workDiv
        = ::alpaka::WorkDivMembers<pmacc::AlpakaDim<DIM1>, pmacc::IdxType>{workExtent, workExtent, workExtent};

    auto input = std::make_unique<pmacc::HostBuffer<int, DIM1>>(one);
    auto device = std::make_unique<pmacc::DeviceBuffer<int, DIM1>>(one, true);
    pmacc::HostBuffer<int, DIM1> output(one);
    input->data()[0] = 41;

    auto step = caravan::alpaka::then(
        caravan::alpaka::then(
            caravan::alpaka::then(
                pmacc::async::fill(queue, device->getOwnedAlpakaView(), 0u),
                pmacc::async::copy(queue, device->getOwnedAlpakaView(), input->getOwnedAlpakaView(), extent)),
            pmacc::async::kernel<pmacc::Acc<DIM1>>(
                queue,
                workDiv,
                Increment{},
                pmacc::async::retain(device->data(), device->getOwnedAlpakaView()))),
        caravan::alpaka::then(
            caravan::alpaka::size(queue, device->sizeOnDeviceBuffer(), device->sizeHostSideBuffer()),
            pmacc::async::copy(queue, output.getOwnedAlpakaView(), device->getOwnedAlpakaView(), extent)));

    pmacc::async::Context context;
    auto completion = context.spawn(std::move(step));
    input.reset();
    device.reset();
    context.wait(completion);

    CHECK(output.data()[0] == 42);
}
