/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <pmacc/fields/Communication.hpp>

#include <array>
#include <utility>

#include <caravan/core.hpp>

namespace picongpu::fields::detail
{
    /** Start an additive guard-to-border scatter for a field after its predecessors.
     *
     * Scatter and gather buffers may alias the same device storage. The field's own
     * per-direction tails do not order the two operations against each other, so
     * both are predecessors here and the returned completion replaces
     * @p scatterTail. The field and the device context must outlive completion.
     */
    template<typename T_Field>
    [[nodiscard]] caravan::Event scatter(
        caravan::ControlContext& context,
        T_Field& field,
        caravan::Event previous,
        caravan::Event& scatterTail,
        caravan::Event& gatherTail)
    {
        std::array dependencies{std::move(previous), scatterTail, gatherTail};
        auto completion = pmacc::fields::spawnCommunication(context, field, caravan::whenAll(dependencies));
        scatterTail = completion;
        return completion;
    }

    /** Start an overwrite border-to-guard gather for a buffer aliasing scattered storage.
     *
     * Both previous communication tails and the supplied producer are predecessors,
     * mirroring scatter(); the returned completion replaces @p gatherTail.
     */
    template<typename T_Buffer>
    [[nodiscard]] caravan::Event gather(
        caravan::ControlContext& context,
        T_Buffer& buffer,
        caravan::Event previous,
        caravan::Event& scatterTail,
        caravan::Event& gatherTail)
    {
        std::array dependencies{std::move(previous), scatterTail, gatherTail};
        auto completion = buffer.spawnCommunication(context, caravan::whenAll(dependencies));
        gatherTail = completion;
        return completion;
    }
} // namespace picongpu::fields::detail
