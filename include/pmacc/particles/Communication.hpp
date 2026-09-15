/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/type/Exchange.hpp"

#include <array>
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

    /** Eager runtime-sized adapter for all particle exchange directions. */
    template<typename T_Particles>
    caravan::Event spawnCommunication(
        caravan::ControlContext& context,
        T_Particles& particles,
        caravan::Event previous = {})
    {
        using HandleGuardRegion = typename T_Particles::HandleGuardRegion;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;
        auto& buffer = particles.getParticlesBuffer();
        auto& device = Environment<>::get().DeviceContext();
        std::vector<caravan::Event> sends;
        std::vector<caravan::Event> receives;
        constexpr auto numExchanges = pmacc::traits::NumberOfExchanges<T_Particles::dim>::value;
        sends.reserve(numExchanges);
        receives.reserve(numExchanges);

        for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
        {
            if(buffer.hasSendExchange(exchange))
            {
                std::array dependencies{previous, buffer.sendCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::sequence(
                            caravan::asSender(caravan::whenAll(dependencies)),
                            sendChunks(particles, exchange))));
                buffer.setSendCompletion(exchange, completion);
                sends.push_back(std::move(completion));
            }
            else
                sends.push_back(context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::sequence(
                            caravan::asSender(previous),
                            HandleNotExchanged{}.handleOutgoingAsync(particles, exchange)))));

            if(buffer.hasReceiveExchange(exchange))
            {
                std::array dependencies{previous, buffer.receiveCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::sequence(
                            caravan::asSender(caravan::whenAll(dependencies)),
                            receiveChunks(particles, exchange))));
                buffer.setReceiveCompletion(exchange, completion);
                receives.push_back(std::move(completion));
            }
            else
                receives.push_back(context.spawn(
                    caravan::alpaka::withDevice(
                        device,
                        caravan::sequence(
                            caravan::asSender(previous),
                            HandleNotExchanged{}.handleIncomingAsync(particles, exchange)))));
        }

        auto received = caravan::whenAll(receives);
        auto filled = context.spawn(
            caravan::alpaka::withDevice(
                device,
                caravan::sequence(caravan::asSender(std::move(received)), particles.fillBorderGapsAsync())));
        sends.push_back(std::move(filled));
        return caravan::whenAll(sends);
    }
} // namespace pmacc::particles
