/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/async/Context.hpp"
#include "pmacc/fields/operations/AddExchangeToBorder.hpp"
#include "pmacc/fields/operations/CopyGuardToExchange.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

#include <array>
#include <vector>

namespace pmacc::fields
{
    /** Exchange and insert a field without legacy polling tasks.
     * @pre field and queue outlive the returned Event.
     */
    template<typename T_Field, typename T_Queue>
    caravan::Event asyncCommunication(
        async::Context& context,
        T_Queue& queue,
        T_Field& field,
        caravan::Event previous = {})
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        std::vector<caravan::Event> branches;
        branches.reserve(traits::NumberOfExchanges<T_Field::dim>::value * 2u);

        for(uint32_t exchange = 1u; exchange < traits::NumberOfExchanges<T_Field::dim>::value; ++exchange)
        {
            if(buffer.hasReceiveExchange(exchange))
            {
                auto receive = buffer.asyncReceive(context, queue, exchange, previous);
                auto insert = context.spawn(
                    caravan::letValue(
                        caravan::asSender(receive.event()),
                        [&buffer, &queue, exchange]
                        {
                            return operations::AddExchangeToBorder{}.sender(queue, buffer, SuperCellSize{}, exchange);
                        }));
                buffer.setReceiveCompletion(exchange, insert);
                branches.push_back(std::move(insert));
            }

            if(buffer.hasSendExchange(exchange))
            {
                std::array dependencies{previous, buffer.sendCompletion(exchange)};
                auto pack = context.spawn(
                    caravan::letValue(
                        caravan::asSender(caravan::whenAll(dependencies)),
                        [&buffer, &queue, exchange]
                        {
                            return operations::CopyGuardToExchange{}.sender(queue, buffer, SuperCellSize{}, exchange);
                        }));
                branches.push_back(buffer.asyncSend(context, queue, exchange, std::move(pack)));
            }
        }
        return caravan::whenAll(branches);
    }
} // namespace pmacc::fields
