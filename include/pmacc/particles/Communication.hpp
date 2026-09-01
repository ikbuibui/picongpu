/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include "pmacc/assert.hpp"
#include "pmacc/async/Context.hpp"
#include "pmacc/traits/NumberOfExchanges.hpp"
#include "pmacc/type/Exchange.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace pmacc::particles
{
    namespace detail
    {
        inline bool forwardTerminal(caravan::Event const& event, caravan::EventSource const& result)
        {
            if(event.state() == caravan::CompletionState::failed)
            {
                result.setFailed(event.error());
                return true;
            }
            if(event.state() == caravan::CompletionState::stopped)
            {
                result.setStopped();
                return true;
            }
            return false;
        }

        template<typename T_Particles, typename T_Queue>
        class SendChunks : public std::enable_shared_from_this<SendChunks<T_Particles, T_Queue>>
        {
        public:
            SendChunks(async::Context& context, T_Queue& queue, T_Particles& particles, uint32_t exchange)
                : context(context)
                , scheduler(context.scheduler())
                , queue(queue)
                , particles(particles)
                , exchange(exchange)
                , maxSize(particles.getParticlesBuffer().getSendExchangeStack(exchange).getMaxParticlesCount())
            {
            }

            caravan::Event run(caravan::Event previous)
            {
                std::array dependencies{std::move(previous), particles.getParticlesBuffer().sendCompletion(exchange)};
                auto completion = context.spawn(caravan::asSender(result.event()));
                startPack(caravan::whenAll(dependencies));
                return completion;
            }

        private:
            void startPack(caravan::Event previous)
            {
                auto self = this->shared_from_this();
                auto packed = context.spawn(
                    caravan::letValue(
                        caravan::asSender(std::move(previous)),
                        [self] { return self->particles.copyGuardToExchangeAsync(self->queue, self->exchange); }));
                watch(std::move(packed), [self](caravan::Event event) { self->afterPack(std::move(event)); });
            }

            void afterPack(caravan::Event event)
            {
                if(forwardTerminal(event, result))
                    return;
                lastSize
                    = particles.getParticlesBuffer().getSendExchangeStack(exchange).getDeviceParticlesCurrentSize();
                PMACC_ASSERT(lastSize <= maxSize);
                auto sent
                    = particles.getParticlesBuffer().asyncSendParticles(context, queue, exchange, std::move(event));
                auto self = this->shared_from_this();
                watch(std::move(sent), [self](caravan::Event completion) { self->afterSend(std::move(completion)); });
            }

            void afterSend(caravan::Event event)
            {
                if(forwardTerminal(event, result))
                    return;
                if(lastSize == maxSize)
                {
                    ++retries;
                    startPack(std::move(event));
                    return;
                }
                if(retries != 0u)
                    std::cerr << "Performance warning: send/receive buffer for species "
                              << T_Particles::FrameType::getName() << " is too small (max: " << maxSize
                              << ", direction: " << exchange << " '" << ExchangeTypeNames{}[exchange]
                              << "', retries: " << retries
                              << "). To remove this warning consider increasing BYTES_EXCHANGE_{X,Y,Z} in "
                                 "memory.param"
                              << std::endl;
                result.setReady();
            }

            template<typename T_Callback>
            void watch(caravan::Event event, T_Callback callback)
            {
                static_cast<void>(event.continueWith(scheduler, std::move(callback)));
            }

            async::Context& context;
            caravan::RunLoopScheduler scheduler;
            T_Queue& queue;
            T_Particles& particles;
            uint32_t exchange;
            size_t maxSize;
            size_t lastSize = 0u;
            size_t retries = 0u;
            caravan::EventSource result;
        };

        template<typename T_Particles, typename T_Queue>
        class ReceiveChunks : public std::enable_shared_from_this<ReceiveChunks<T_Particles, T_Queue>>
        {
        public:
            ReceiveChunks(async::Context& context, T_Queue& queue, T_Particles& particles, uint32_t exchange)
                : context(context)
                , scheduler(context.scheduler())
                , queue(queue)
                , particles(particles)
                , exchange(exchange)
                , maxSize(particles.getParticlesBuffer().getReceiveExchangeStack(exchange).getMaxParticlesCount())
            {
            }

            caravan::Event run(caravan::Event previous)
            {
                auto completion = context.spawn(caravan::asSender(result.event()));
                startReceive(std::move(previous));
                return completion;
            }

        private:
            void startReceive(caravan::Event previous)
            {
                auto received = particles.getParticlesBuffer()
                                    .asyncReceiveParticles(context, queue, exchange, std::move(previous));
                auto self = this->shared_from_this();
                watch(
                    std::move(received),
                    [self](caravan::Event completion) { self->afterReceive(std::move(completion)); });
            }

            void afterReceive(caravan::Event event)
            {
                if(forwardTerminal(event, result))
                    return;
                lastSize
                    = particles.getParticlesBuffer().getReceiveExchangeStack(exchange).getHostParticlesCurrentSize();
                PMACC_ASSERT(lastSize <= maxSize);
                if(lastSize == 0u)
                {
                    result.setReady();
                    return;
                }

                auto self = this->shared_from_this();
                auto inserted = context.spawn(
                    caravan::letValue(
                        caravan::asSender(std::move(event)),
                        [self]
                        {
                            return self->particles.insertParticlesAsync(self->queue, self->exchange, self->lastSize);
                        }));
                particles.getParticlesBuffer().setReceiveCompletion(exchange, inserted);
                watch(
                    std::move(inserted),
                    [self](caravan::Event completion) { self->afterInsert(std::move(completion)); });
            }

            void afterInsert(caravan::Event event)
            {
                if(forwardTerminal(event, result))
                    return;
                if(lastSize == maxSize)
                    startReceive(std::move(event));
                else
                    result.setReady();
            }

            template<typename T_Callback>
            void watch(caravan::Event event, T_Callback callback)
            {
                static_cast<void>(event.continueWith(scheduler, std::move(callback)));
            }

            async::Context& context;
            caravan::RunLoopScheduler scheduler;
            T_Queue& queue;
            T_Particles& particles;
            uint32_t exchange;
            size_t maxSize;
            size_t lastSize = 0u;
            caravan::EventSource result;
        };
    } // namespace detail

    /** Exchange all particle chunks and fill received border gaps without polling tasks.
     * @pre particles and queue outlive the returned Event.
     */
    template<typename T_Particles, typename T_Queue>
    caravan::Event asyncCommunication(
        async::Context& context,
        T_Queue& queue,
        T_Particles& particles,
        caravan::Event previous = {})
    {
        using HandleGuardRegion = typename T_Particles::HandleGuardRegion;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;
        std::vector<caravan::Event> sends;
        std::vector<caravan::Event> receives;
        constexpr auto numExchanges = traits::NumberOfExchanges<T_Particles::dim>::value;
        sends.reserve(numExchanges);
        receives.reserve(numExchanges);

        for(uint32_t exchange = 1u; exchange < numExchanges; ++exchange)
        {
            if(particles.getParticlesBuffer().hasSendExchange(exchange))
            {
                auto state
                    = std::make_shared<detail::SendChunks<T_Particles, T_Queue>>(context, queue, particles, exchange);
                sends.push_back(state->run(previous));
            }
            else
                sends.push_back(context.spawn(
                    caravan::letValue(
                        caravan::asSender(previous),
                        [&queue, &particles, exchange]
                        { return HandleNotExchanged{}.handleOutgoingAsync(queue, particles, exchange); })));

            if(particles.getParticlesBuffer().hasReceiveExchange(exchange))
            {
                auto state = std::make_shared<detail::ReceiveChunks<T_Particles, T_Queue>>(
                    context,
                    queue,
                    particles,
                    exchange);
                receives.push_back(state->run(previous));
            }
            else
                receives.push_back(context.spawn(
                    caravan::letValue(
                        caravan::asSender(previous),
                        [&queue, &particles, exchange]
                        { return HandleNotExchanged{}.handleIncomingAsync(queue, particles, exchange); })));
        }

        auto received = caravan::whenAll(receives);
        auto filled = context.spawn(
            caravan::letValue(
                caravan::asSender(std::move(received)),
                [&queue, &particles] { return particles.fillBorderGapsAsync(queue); }));
        sends.push_back(std::move(filled));
        return caravan::whenAll(sends);
    }
} // namespace pmacc::particles
