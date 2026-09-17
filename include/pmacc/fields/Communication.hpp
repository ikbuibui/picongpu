/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/fields/operations/AddExchangeToBorder.hpp"
#include "pmacc/fields/operations/CopyGuardToExchange.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"

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
               | caravan::sequence(operations::AddExchangeToBorder{}.sender(buffer, SuperCellSize{}, exchange));
    }

    /** Describe one lazy field pack/send branch. */
    template<typename T_Field>
    auto sendExchange(T_Field& field, uint32_t exchange)
    {
        using SuperCellSize = typename T_Field::MappingDesc::SuperCellSize;
        auto& buffer = field.getGridBuffer();
        return operations::CopyGuardToExchange{}.sender(buffer, SuperCellSize{}, exchange)
               | caravan::sequence(buffer.send(exchange));
    }

    /** Describe additive guard-to-border field communication as one lazy aggregate. */
    template<typename T_Field>
    auto communication(T_Field& field, caravan::Event previous = {})
    {
        return caravan::deferWithReservations(
            [&field, previous = std::move(previous)](caravan::EventReservations& reservations) mutable
            {
                auto& buffer = field.getGridBuffer();
                auto& device = Environment<>::get().DeviceContext();
                auto receivePrevious = previous;
                // Direction events are rolled back if a later direction fails to connect.
                auto makeReceive = [&field, &buffer, &device, &receivePrevious, &reservations](uint32_t exchange)
                {
                    auto reuse = buffer.receiveCompletion(exchange);
                    caravan::EventSource completion;
                    auto result = completion.event();
                    reservations.replace(
                        reuse,
                        result,
                        [&buffer, exchange](caravan::Event event) noexcept
                        { buffer.setReceiveCompletion(exchange, std::move(event)); });
                    auto branch = caravan::alpaka::withDevice(
                        device,
                        caravan::whenAll(
                            caravan::asSender(receivePrevious),
                            caravan::asSender(std::move(reuse)))
                            | caravan::sequence(receiveExchange(field, exchange)));
                    receivePrevious = result;
                    return caravan::trackCompletion(std::move(branch), std::move(completion));
                };
                auto makeSend = [&field, &buffer, &device, previous, &reservations](uint32_t exchange)
                {
                    auto reuse = buffer.sendCompletion(exchange);
                    caravan::EventSource completion;
                    reservations.replace(
                        reuse,
                        completion.event(),
                        [&buffer, exchange](caravan::Event event) noexcept
                        { buffer.setSendCompletion(exchange, std::move(event)); });
                    auto branch = caravan::alpaka::withDevice(
                        device,
                        caravan::whenAll(caravan::asSender(previous), caravan::asSender(std::move(reuse)))
                            | caravan::sequence(sendExchange(field, exchange)));
                    return caravan::trackCompletion(std::move(branch), std::move(completion));
                };

                using ReceiveBranch = decltype(makeReceive(1u));
                using SendBranch = decltype(makeSend(1u));
                std::vector<ReceiveBranch> receives;
                std::vector<SendBranch> sends;
                constexpr auto numExchanges = traits::NumberOfExchanges<T_Field::dim>::value;
                receives.reserve(numExchanges);
                sends.reserve(numExchanges);
                for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
                {
                    if(buffer.hasReceiveExchange(exchange))
                        receives.push_back(makeReceive(exchange));
                    if(buffer.hasSendExchange(exchange))
                        sends.push_back(makeSend(exchange));
                }
                return caravan::whenAll(
                    caravan::asSender(std::move(previous)),
                    caravan::whenAll(std::move(receives)),
                    caravan::whenAll(std::move(sends)));
            });
    }
} // namespace pmacc::fields
