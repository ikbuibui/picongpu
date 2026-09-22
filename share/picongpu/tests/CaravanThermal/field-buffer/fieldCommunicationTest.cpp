/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 *
 * Focused test for the production scatter/gather communication composition used
 * by FieldTmp. It uses real GridBuffers, the same exchange setup as PMacc's own
 * field-communication test, and aliased scatter/gather storage.
 */
#include <picongpu/fields/detail/FieldCommunicationOperations.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/fields/Communication.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/traits/GetUniqueTypeId.hpp>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <vector>

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

        pmacc::GridBuffer<int, TEST_DIM> buffer{
            pmacc::GridLayout<TEST_DIM>{
                pmacc::DataSpace<TEST_DIM>::create(2),
                pmacc::DataSpace<TEST_DIM>::create(1)}};
    };

    using Buffer = pmacc::GridBuffer<int, TEST_DIM>;

    /** Add the additive guard-to-border exchanges used by the scatter path. */
    void addAdditiveExchanges(MockField& field)
    {
        auto& buffer = field.getGridBuffer();
        auto const tag = pmacc::traits::getUniqueId<uint32_t>();
        for(uint32_t exchange = 1u; exchange < pmacc::traits::NumberOfExchanges<TEST_DIM>::value; ++exchange)
        {
            auto const direction = pmacc::Mask::getRelativeDirections<TEST_DIM>(exchange);
            auto extent = pmacc::DataSpace<TEST_DIM>::create(1);
            for(uint32_t d = 0u; d < TEST_DIM; ++d)
                if(direction[d] == 0)
                    extent[d] = 2;
            buffer.addExchangeBuffer(exchange, extent, tag);
        }
    }

    /** Add the overwrite border-to-guard exchanges used by the gather path. */
    void addOverwriteExchanges(Buffer& buffer)
    {
        auto const tag = pmacc::traits::getUniqueId<uint32_t>();
        for(uint32_t exchange = 1u; exchange < pmacc::traits::NumberOfExchanges<TEST_DIM>::value; ++exchange)
        {
            auto const extent = pmacc::DataSpace<TEST_DIM>::create(1);
            buffer.addExchange(pmacc::GUARD, exchange, extent, tag);
        }
    }

    bool isGuard(pmacc::DataSpace<TEST_DIM> const& cell)
    {
        for(uint32_t d = 0u; d < TEST_DIM; ++d)
            if(cell[d] == 0 || cell[d] == 3)
                return true;
        return false;
    }

    /** Initialize valid device data before scheduling communication. */
    void fillAndUpload(Buffer& buffer, int const base)
    {
        auto* const data = buffer.getHostBuffer().data();
        auto const capacity = buffer.getHostBuffer().capacityND().productOfComponents();
        for(std::size_t i = 0u; i < capacity; ++i)
            data[i] = base + static_cast<int>(i);

        auto& device = pmacc::Environment<>::get().DeviceContext();
        caravan::ControlContext context;
        context.wait(context.spawn(caravan::alpaka::withDevice(device, buffer.hostToDevice())));
    }

    /** Release every pending test gate on scope exit so a fatal assertion cannot
     * leave the owned ControlContext joining work that never completes.
     */
    struct ReleaseGates
    {
        caravan::EventSource& first;
        caravan::EventSource& second;
        caravan::EventSource& third;

        ~ReleaseGates()
        {
            first.setReady();
            second.setReady();
            third.setReady();
        }
    };

    enum class PendingGate
    {
        Producer,
        ScatterTail,
        GatherTail
    };

    /** Run scatter -> gather -> scatter for two steps on an aliased pair.
     *
     * @param pipelined true chains the operations through one shared completion;
     *                  false waits after every operation. Both must yield the same
     *                  values if the dependency composition is correct.
     */
    std::vector<int> runAliasedSequence(bool const pipelined)
    {
        MockField scatterField;
        addAdditiveExchanges(scatterField);
        auto& scatterBuffer = scatterField.getGridBuffer();
        Buffer gatherBuffer{scatterBuffer.getDeviceBuffer(), scatterBuffer.getGridLayout()};
        addOverwriteExchanges(gatherBuffer);

        auto& device = pmacc::Environment<>::get().DeviceContext();
        auto const extent = scatterBuffer.getGridLayout().sizeND();
        auto box = scatterBuffer.getHostBuffer().getDataBox();
        auto const count = extent.productOfComponents();

        caravan::ControlContext context;
        caravan::Event scatterTail;
        caravan::Event gatherTail;

        for(int step = 0; step < 2; ++step)
        {
            for(int i = 0; i < count; ++i)
            {
                auto const cell = pmacc::math::mapToND(extent, i);
                box(cell) = isGuard(cell) ? 1 : 10;
            }
            context.wait(context.spawn(caravan::alpaka::withDevice(device, scatterBuffer.hostToDevice())));

            // Only the stored tails order the alternating operations; no external
            // producer or chained completion hides a missing tail dependency.
            if(pipelined)
            {
                // The helper stores each completion in the tails, so only the last
                // event needs to be waited on; the earlier ones remain as predecessors.
                [[maybe_unused]] auto const scatterFirst
                    = picongpu::fields::detail::scatter(context, scatterField, {}, scatterTail, gatherTail);
                [[maybe_unused]] auto const gathered
                    = picongpu::fields::detail::gather(context, gatherBuffer, {}, scatterTail, gatherTail);
                auto const completion
                    = picongpu::fields::detail::scatter(context, scatterField, {}, scatterTail, gatherTail);
                context.wait(completion);
            }
            else
            {
                context.wait(picongpu::fields::detail::scatter(context, scatterField, {}, scatterTail, gatherTail));
                context.wait(picongpu::fields::detail::gather(context, gatherBuffer, {}, scatterTail, gatherTail));
                context.wait(picongpu::fields::detail::scatter(context, scatterField, {}, scatterTail, gatherTail));
            }

            context.wait(context.spawn(caravan::alpaka::withDevice(device, scatterBuffer.deviceToHost())));
        }

        std::vector<int> result(count);
        for(int i = 0; i < count; ++i)
            result[i] = box(pmacc::math::mapToND(extent, i));
        return result;
    }
} // namespace

TEST_CASE("FieldTmp scatter depends independently on producer and both tails", "[picongpu][fields]")
{
    PendingGate const pendingGate = GENERATE(
        PendingGate::Producer,
        PendingGate::ScatterTail,
        PendingGate::GatherTail);

    MockField field;
    addAdditiveExchanges(field);
    fillAndUpload(field.getGridBuffer(), 3);

    caravan::ControlContext context;
    caravan::EventSource producer;
    caravan::EventSource scatterTailSource;
    caravan::EventSource gatherTailSource;
    ReleaseGates const releaseGates{producer, scatterTailSource, gatherTailSource};

    caravan::Event scatterTail = scatterTailSource.event();
    caravan::Event gatherTail = gatherTailSource.event();

    // Exactly one dependency stays pending; the other two are already ready.
    if(pendingGate != PendingGate::Producer)
        producer.setReady();
    if(pendingGate != PendingGate::ScatterTail)
        scatterTailSource.setReady();
    if(pendingGate != PendingGate::GatherTail)
        gatherTailSource.setReady();

    auto completion = picongpu::fields::detail::scatter(context, field, producer.event(), scatterTail, gatherTail);

    context.runReady();
    CHECK_FALSE(completion.isReady());

    switch(pendingGate)
    {
    case PendingGate::Producer:
        producer.setReady();
        break;
    case PendingGate::ScatterTail:
        scatterTailSource.setReady();
        break;
    case PendingGate::GatherTail:
        gatherTailSource.setReady();
        break;
    }
    context.wait(completion);
    CHECK(completion.isReady());
}

TEST_CASE("FieldTmp gather depends independently on producer and both tails", "[picongpu][fields]")
{
    PendingGate const pendingGate = GENERATE(
        PendingGate::Producer,
        PendingGate::ScatterTail,
        PendingGate::GatherTail);

    MockField field;
    addOverwriteExchanges(field.getGridBuffer());
    fillAndUpload(field.getGridBuffer(), 5);

    caravan::ControlContext context;
    caravan::EventSource producer;
    caravan::EventSource scatterTailSource;
    caravan::EventSource gatherTailSource;
    ReleaseGates const releaseGates{producer, scatterTailSource, gatherTailSource};

    caravan::Event scatterTail = scatterTailSource.event();
    caravan::Event gatherTail = gatherTailSource.event();

    if(pendingGate != PendingGate::Producer)
        producer.setReady();
    if(pendingGate != PendingGate::ScatterTail)
        scatterTailSource.setReady();
    if(pendingGate != PendingGate::GatherTail)
        gatherTailSource.setReady();

    auto completion
        = picongpu::fields::detail::gather(context, field.getGridBuffer(), producer.event(), scatterTail, gatherTail);

    context.runReady();
    CHECK_FALSE(completion.isReady());

    switch(pendingGate)
    {
    case PendingGate::Producer:
        producer.setReady();
        break;
    case PendingGate::ScatterTail:
        scatterTailSource.setReady();
        break;
    case PendingGate::GatherTail:
        gatherTailSource.setReady();
        break;
    }
    context.wait(completion);
    CHECK(completion.isReady());
}

TEST_CASE("FieldTmp communication without exchanges preserves each predecessor", "[picongpu][fields]")
{
    enum class Helper
    {
        Scatter,
        Gather
    };

    auto const helper = GENERATE(Helper::Scatter, Helper::Gather);
    auto const pendingGate = GENERATE(
        PendingGate::Producer,
        PendingGate::ScatterTail,
        PendingGate::GatherTail);

    MockField field; // no exchanges configured

    caravan::ControlContext context;
    caravan::EventSource producer;
    caravan::EventSource scatterTailSource;
    caravan::EventSource gatherTailSource;
    ReleaseGates const releaseGates{producer, scatterTailSource, gatherTailSource};

    caravan::Event scatterTail = scatterTailSource.event();
    caravan::Event gatherTail = gatherTailSource.event();

    if(pendingGate != PendingGate::Producer)
        producer.setReady();
    if(pendingGate != PendingGate::ScatterTail)
        scatterTailSource.setReady();
    if(pendingGate != PendingGate::GatherTail)
        gatherTailSource.setReady();

    auto completion = helper == Helper::Scatter
                          ? picongpu::fields::detail::scatter(
                                context,
                                field,
                                producer.event(),
                                scatterTail,
                                gatherTail)
                          : picongpu::fields::detail::gather(
                                context,
                                field.getGridBuffer(),
                                producer.event(),
                                scatterTail,
                                gatherTail);

    context.runReady();
    CHECK_FALSE(completion.isReady());

    switch(pendingGate)
    {
    case PendingGate::Producer:
        producer.setReady();
        break;
    case PendingGate::ScatterTail:
        scatterTailSource.setReady();
        break;
    case PendingGate::GatherTail:
        gatherTailSource.setReady();
        break;
    }
    context.wait(completion);
    CHECK(completion.isReady());
}

TEST_CASE("FieldTmp alternating communication on aliased storage matches serialized execution", "[picongpu][fields]")
{
    auto const pipelined = runAliasedSequence(true);
    auto const serialized = runAliasedSequence(false);

    REQUIRE(pipelined.size() == serialized.size());
    for(std::size_t i = 0u; i < pipelined.size(); ++i)
        CHECK(pipelined[i] == serialized[i]);
}

TEST_CASE("FieldTmp aliased scatter/gather produces expected intermediate values", "[picongpu][fields]")
{
    MockField scatterField;
    addAdditiveExchanges(scatterField);
    auto& scatterBuffer = scatterField.getGridBuffer();
    Buffer gatherBuffer{scatterBuffer.getDeviceBuffer(), scatterBuffer.getGridLayout()};
    addOverwriteExchanges(gatherBuffer);

    auto& device = pmacc::Environment<>::get().DeviceContext();
    auto const extent = scatterBuffer.getGridLayout().sizeND();
    auto box = scatterBuffer.getHostBuffer().getDataBox();
    auto const count = extent.productOfComponents();

    caravan::ControlContext context;
    caravan::Event scatterTail;
    caravan::Event gatherTail;

    for(int i = 0; i < count; ++i)
    {
        auto const cell = pmacc::math::mapToND(extent, i);
        box(cell) = isGuard(cell) ? 1 : 10;
    }
    context.wait(context.spawn(caravan::alpaka::withDevice(device, scatterBuffer.hostToDevice())));

    auto const contributions = (1 << TEST_DIM) - 1;
    auto const interiorScatter = 10 + contributions;
    auto checkValues = [&](int const interior, int const guard)
    {
        for(int i = 0; i < count; ++i)
        {
            auto const cell = pmacc::math::mapToND(extent, i);
            CHECK(box(cell) == (isGuard(cell) ? guard : interior));
        }
    };
    auto download = [&]
    { context.wait(context.spawn(caravan::alpaka::withDevice(device, scatterBuffer.deviceToHost()))); };

    // Scatter adds one contribution per guard direction into the interior.
    context.wait(picongpu::fields::detail::scatter(context, scatterField, {}, scatterTail, gatherTail));
    download();
    checkValues(interiorScatter, 1);

    // Gather overwrites guards with the interior value of the neighboring border.
    context.wait(picongpu::fields::detail::gather(context, gatherBuffer, {}, scatterTail, gatherTail));
    download();
    checkValues(interiorScatter, interiorScatter);

    // The second scatter adds the now-larger guard values again.
    context.wait(picongpu::fields::detail::scatter(context, scatterField, {}, scatterTail, gatherTail));
    download();
    checkValues(interiorScatter * (contributions + 1), interiorScatter);
}
