/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/fields/operations/AddExchangeToBorder.hpp"
#include "pmacc/fields/operations/CopyGuardToExchange.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

#include <array>
#include <vector>

#include <caravan/core.hpp>

namespace pmacc::fields
{
    /** Describe one lazy field receive/insert branch. */
    template<typename T_Field, typename T_Queue>
    auto receiveExchange(T_Queue& queue, T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return buffer.receive(queue, exchange)
               | caravan::letValue(
                   [&buffer, &queue, exchange](auto const&)
                   { return operations::AddExchangeToBorder{}.sender(queue, buffer, SuperCellSize{}, exchange); });
    }

    /** Describe one lazy field pack/send branch. */
    template<typename T_Field, typename T_Queue>
    auto sendExchange(T_Queue& queue, T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return operations::CopyGuardToExchange{}.sender(queue, buffer, SuperCellSize{}, exchange)
               | caravan::letValue([&buffer, &queue, exchange] { return buffer.send(queue, exchange); });
    }

    /**
     * Eager runtime-sized adapter for field communication.
     * @pre field and queue outlive the returned Event.
     */
    template<typename T_Field, typename T_Queue>
    caravan::Event spawnCommunication(
        caravan::ControlContext& context,
        T_Queue& queue,
        T_Field& field,
        caravan::Event previous = {})
    {
        auto& buffer = field.getGridBuffer();
        std::vector<caravan::Event> branches;
        branches.reserve(traits::NumberOfExchanges<T_Field::dim>::value * 2u);

        for(uint32_t exchange = 1u; exchange < traits::NumberOfExchanges<T_Field::dim>::value; ++exchange)
        {
            if(buffer.hasReceiveExchange(exchange))
            {
                std::array dependencies{previous, buffer.receiveCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::asSender(caravan::whenAll(dependencies))
                    | caravan::letValue([&queue, &field, exchange]
                                        { return receiveExchange(queue, field, exchange); }));
                buffer.setReceiveCompletion(exchange, completion);
                branches.push_back(std::move(completion));
            }

            if(buffer.hasSendExchange(exchange))
            {
                std::array dependencies{previous, buffer.sendCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::asSender(caravan::whenAll(dependencies))
                    | caravan::letValue([&queue, &field, exchange] { return sendExchange(queue, field, exchange); }));
                buffer.setSendCompletion(exchange, completion);
                branches.push_back(std::move(completion));
            }
        }
        return caravan::whenAll(branches);
    }
} // namespace pmacc::fields
