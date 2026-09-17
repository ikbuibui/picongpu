/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/type/Exchange.hpp"

#include <exception>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/core.hpp>

namespace pmacc::particles
{
    namespace detail
    {
        /** Skip empty chunks without constructing or launching a zero-sized insertion kernel. */
        template<typename T_Particles>
        struct InsertNonEmptySender
        {
            using completion_signatures = caravan::
                CompletionSignatures<caravan::ValueSignature<>, caravan::ErrorSignature<std::exception_ptr>>;

            template<typename T_Receiver>
            class Operation
            {
                using InsertSender = decltype(std::declval<T_Particles&>().insertParticlesAsync(
                    std::declval<uint32_t>(),
                    std::declval<size_t>()));

            public:
                Operation(InsertNonEmptySender sender, T_Receiver receiver)
                    : m_sender(sender)
                    , m_receiver(std::move(receiver))
                {
                }

                Operation(Operation const&) = delete;
                Operation& operator=(Operation const&) = delete;
                Operation(Operation&&) = delete;
                Operation& operator=(Operation&&) = delete;

                void start() & noexcept
                {
                    if(m_sender.count == 0u)
                    {
                        m_receiver.set_value();
                        return;
                    }
                    try
                    {
                        m_insert.emplace(
                            m_sender.particles.insertParticlesAsync(m_sender.exchange, m_sender.count),
                            m_receiver);
                        m_insert->start();
                    }
                    catch(...)
                    {
                        m_receiver.set_error(std::current_exception());
                    }
                }

            private:
                InsertNonEmptySender m_sender;
                T_Receiver m_receiver;
                std::optional<caravan::detail::ConnectedOperation<InsertSender, T_Receiver>> m_insert;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{*this, std::forward<T_Receiver>(receiver)};
            }

            T_Particles& particles;
            uint32_t exchange;
            size_t count;
        };
    } // namespace detail

    template<typename T_Particles>
    auto sendChunks(T_Particles& particles, uint32_t exchange)
    {
        return caravan::repeatUntil(
            [&particles, exchange, retries = size_t{0u}, lastSize = size_t{0u}]() mutable
            {
                auto const maxSize
                    = particles.getParticlesBuffer().getSendExchangeStack(exchange).getMaxParticlesCount();
                return particles.copyGuardToExchangeAsync(exchange)
                       | caravan::letValue(
                           [&particles, exchange, maxSize, &lastSize]
                           {
                               lastSize = particles.getParticlesBuffer()
                                              .getSendExchangeStack(exchange)
                                              .getDeviceParticlesCurrentSize();
                               PMACC_ASSERT(lastSize <= maxSize);
                               return particles.getParticlesBuffer().sendParticles(exchange);
                           })
                       | caravan::then(
                           [exchange, maxSize, &lastSize, &retries](auto&&...)
                           {
                               if(lastSize == maxSize)
                               {
                                   ++retries;
                                   return false;
                               }
                               if(retries != 0u)
                                   std::cerr << "Performance warning: send/receive buffer for species "
                                             << T_Particles::FrameType::getName() << " is too small (max: " << maxSize
                                             << ", direction: " << exchange << " '" << ExchangeTypeNames{}[exchange]
                                             << "', retries: " << retries
                                             << "). To remove this warning consider increasing BYTES_EXCHANGE_{X,Y,Z} "
                                                "in memory.param"
                                             << std::endl;
                               return true;
                           });
            });
    }

    template<typename T_Particles>
    auto receiveChunks(T_Particles& particles, uint32_t exchange)
    {
        return caravan::repeatUntil(
            [&particles, exchange]
            {
                auto const maxSize
                    = particles.getParticlesBuffer().getReceiveExchangeStack(exchange).getMaxParticlesCount();
                return particles.getParticlesBuffer().receiveParticles(exchange)
                       | caravan::letValue(
                           [&particles, exchange, maxSize]
                           {
                               auto const lastSize = particles.getParticlesBuffer()
                                                         .getReceiveExchangeStack(exchange)
                                                         .getHostParticlesCurrentSize();
                               PMACC_ASSERT(lastSize <= maxSize);
                               return detail::InsertNonEmptySender<T_Particles>{particles, exchange, lastSize}
                                      | caravan::then([lastSize, maxSize]
                                                      { return lastSize == 0u || lastSize < maxSize; });
                           });
            });
    }

    /** Describe all particle exchange directions as one lazy aggregate. */
    template<typename T_Particles>
    auto communication(T_Particles& particles, caravan::Event previous = {})
    {
        using HandleGuardRegion = typename T_Particles::HandleGuardRegion;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;
        return caravan::defer(
            [&particles, previous = std::move(previous)]() mutable
            {
                auto& buffer = particles.getParticlesBuffer();
                auto& device = Environment<>::get().DeviceContext();
                auto makeSend = [&particles, &buffer, &device, previous](uint32_t exchange)
                {
                    auto reuse = buffer.sendCompletion(exchange);
                    caravan::EventSource completion;
                    buffer.setSendCompletion(exchange, completion.event());
                    auto branch = caravan::alpaka::withDevice(
                        device,
                        caravan::whenAll(caravan::asSender(previous), caravan::asSender(std::move(reuse)))
                            | caravan::sequence(sendChunks(particles, exchange)));
                    return caravan::trackCompletion(std::move(branch), std::move(completion));
                };
                auto makeReceive = [&particles, &buffer, &device, previous](uint32_t exchange)
                {
                    auto reuse = buffer.receiveCompletion(exchange);
                    caravan::EventSource completion;
                    buffer.setReceiveCompletion(exchange, completion.event());
                    auto branch = caravan::alpaka::withDevice(
                        device,
                        caravan::whenAll(caravan::asSender(previous), caravan::asSender(std::move(reuse)))
                            | caravan::sequence(receiveChunks(particles, exchange)));
                    return caravan::trackCompletion(std::move(branch), std::move(completion));
                };
                auto makeInactiveSend = [&particles, &device, previous](uint32_t exchange)
                {
                    return caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(previous)
                            | caravan::sequence(HandleNotExchanged{}.handleOutgoingAsync(particles, exchange)));
                };
                auto makeInactiveReceive = [&particles, &device, previous](uint32_t exchange)
                {
                    return caravan::alpaka::withDevice(
                        device,
                        caravan::asSender(previous)
                            | caravan::sequence(HandleNotExchanged{}.handleIncomingAsync(particles, exchange)));
                };

                using SendBranch = decltype(makeSend(1u));
                using ReceiveBranch = decltype(makeReceive(1u));
                using InactiveSendBranch = decltype(makeInactiveSend(1u));
                using InactiveReceiveBranch = decltype(makeInactiveReceive(1u));
                std::vector<SendBranch> sends;
                std::vector<ReceiveBranch> receives;
                std::vector<InactiveSendBranch> inactiveSends;
                std::vector<InactiveReceiveBranch> inactiveReceives;
                constexpr auto numExchanges = pmacc::traits::NumberOfExchanges<T_Particles::dim>::value;
                sends.reserve(numExchanges);
                receives.reserve(numExchanges);
                inactiveSends.reserve(numExchanges);
                inactiveReceives.reserve(numExchanges);

                for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
                {
                    if(buffer.hasSendExchange(exchange))
                        sends.push_back(makeSend(exchange));
                    else
                        inactiveSends.push_back(makeInactiveSend(exchange));

                    if(buffer.hasReceiveExchange(exchange))
                        receives.push_back(makeReceive(exchange));
                    else
                        inactiveReceives.push_back(makeInactiveReceive(exchange));
                }

                auto received = caravan::whenAll(
                    caravan::whenAll(std::move(receives)),
                    caravan::whenAll(std::move(inactiveReceives)));
                auto filled = caravan::alpaka::withDevice(
                    device,
                    std::move(received) | caravan::sequence(particles.fillBorderGapsAsync()));
                return caravan::whenAll(
                    caravan::asSender(std::move(previous)),
                    caravan::whenAll(std::move(sends)),
                    caravan::whenAll(std::move(inactiveSends)),
                    std::move(filled));
            });
    }
} // namespace pmacc::particles
