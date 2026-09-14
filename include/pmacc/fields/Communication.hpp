/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/fields/operations/AddExchangeToBorder.hpp"
#include "pmacc/fields/operations/CopyGuardToExchange.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

#include <array>
#include <vector>

#include <caravan/core.hpp>

namespace pmacc::fields
{
    /** Describe one lazy field receive/insert branch. */
    template<typename T_Field>
    auto receiveExchange(T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return buffer.receive(exchange)
               | caravan::letValue(
                   [&buffer, exchange](auto const&)
                   { return operations::AddExchangeToBorder{}.sender(buffer, SuperCellSize{}, exchange); });
    }

    /** Describe one lazy field pack/send branch. */
    template<typename T_Field>
    auto sendExchange(T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return operations::CopyGuardToExchange{}.sender(buffer, SuperCellSize{}, exchange)
               | caravan::letValue([&buffer, exchange] { return buffer.send(exchange); });
    }

    /** Eager runtime-sized adapter for field communication. */
    template<typename T_Field>
    caravan::Event spawnCommunication(caravan::ControlContext& context, T_Field& field, caravan::Event previous = {})
    {
        auto& buffer = field.getGridBuffer();
        auto& device = Environment<>::get().DeviceContext();
        std::vector<caravan::Event> branches;
        branches.reserve(traits::NumberOfExchanges<T_Field::dim>::value * 2u);
        auto receivePrevious = previous;

        for(uint32_t exchange = 1u; exchange < traits::NumberOfExchanges<T_Field::dim>::value; ++exchange)
        {
            if(buffer.hasReceiveExchange(exchange))
            {
                // Inserts use += and neighboring directions overlap at edges; retain their former FIFO ordering.
                std::array dependencies{receivePrevious, buffer.receiveCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(caravan::whenAll(dependencies))
                            | caravan::letValue([&field, exchange] { return receiveExchange(field, exchange); })));
                buffer.setReceiveCompletion(exchange, completion);
                receivePrevious = completion;
                branches.push_back(std::move(completion));
            }

            if(buffer.hasSendExchange(exchange))
            {
                std::array dependencies{previous, buffer.sendCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(caravan::whenAll(dependencies))
                            | caravan::letValue([&field, exchange] { return sendExchange(field, exchange); })));
                buffer.setSendCompletion(exchange, completion);
                branches.push_back(std::move(completion));
            }
        }
        return caravan::whenAll(branches);
    }
} // namespace pmacc::fields
