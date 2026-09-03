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
#include <exception>
#include <iostream>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace pmacc::particles
{
    namespace detail
    {
        template<typename T_Particles, typename T_Queue, typename T_Receiver>
        class SendChunksOperation
        {
            struct PackReceiver
            {
                template<typename... T>
                void set_value(T&&...) noexcept
                {
                    owner->afterPack();
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->fail(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->stop();
                }

                SendChunksOperation* owner;
            };

            struct SendReceiver
            {
                template<typename... T>
                void set_value(T&&...) noexcept
                {
                    owner->afterSend();
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->fail(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->stop();
                }

                SendChunksOperation* owner;
            };

            using PackSender = decltype(std::declval<T_Particles&>().copyGuardToExchangeAsync(
                std::declval<T_Queue&>(),
                std::declval<uint32_t>()));
            using SendSender = decltype(std::declval<T_Particles&>().getParticlesBuffer().sendParticles(
                std::declval<T_Queue&>(),
                std::declval<uint32_t>()));

            class PackOperation
            {
            public:
                PackOperation(PackSender sender, SendChunksOperation* owner)
                    : operation(std::move(sender).connect(PackReceiver{owner}))
                {
                }

                void start() noexcept
                {
                    operation.start();
                }

            private:
                decltype(std::declval<PackSender&&>().connect(std::declval<PackReceiver>())) operation;
            };

            class SendOperation
            {
            public:
                SendOperation(SendSender sender, SendChunksOperation* owner)
                    : operation(std::move(sender).connect(SendReceiver{owner}))
                {
                }

                void start() noexcept
                {
                    operation.start();
                }

            private:
                decltype(std::declval<SendSender&&>().connect(std::declval<SendReceiver>())) operation;
            };

        public:
            SendChunksOperation(T_Queue& queue, T_Particles& particles, uint32_t exchange, T_Receiver receiver)
                : queue(queue)
                , particles(particles)
                , exchange(exchange)
                , receiver(std::move(receiver))
            {
            }

            SendChunksOperation(SendChunksOperation const&) = delete;
            SendChunksOperation& operator=(SendChunksOperation const&) = delete;
            SendChunksOperation(SendChunksOperation&&) = delete;
            SendChunksOperation& operator=(SendChunksOperation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    maxSize = particles.getParticlesBuffer().getSendExchangeStack(exchange).getMaxParticlesCount();
                    startPack();
                }
                catch(...)
                {
                    fail(std::current_exception());
                }
            }

        private:
            void startPack()
            {
                packs.emplace_back(particles.copyGuardToExchangeAsync(queue, exchange), this);
                packs.back().start();
            }

            void afterPack() noexcept
            {
                try
                {
                    lastSize = particles.getParticlesBuffer()
                                   .getSendExchangeStack(exchange)
                                   .getDeviceParticlesCurrentSize();
                    PMACC_ASSERT(lastSize <= maxSize);
                    sends.emplace_back(particles.getParticlesBuffer().sendParticles(queue, exchange), this);
                    sends.back().start();
                }
                catch(...)
                {
                    fail(std::current_exception());
                }
            }

            void afterSend() noexcept
            {
                try
                {
                    if(lastSize == maxSize)
                    {
                        ++retries;
                        startPack();
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
                    receiver.set_value();
                }
                catch(...)
                {
                    fail(std::current_exception());
                }
            }

            void fail(std::exception_ptr error) noexcept
            {
                receiver.set_error(std::move(error));
            }

            void stop() noexcept
            {
                receiver.set_stopped();
            }

            T_Queue& queue;
            T_Particles& particles;
            uint32_t exchange;
            T_Receiver receiver;
            size_t maxSize = 0u;
            size_t lastSize = 0u;
            size_t retries = 0u;
            // ponytail: retain stage states for synchronous completion; reuse slots if P1 shows this allocation
            // matters.
            std::list<PackOperation> packs;
            std::list<SendOperation> sends;
        };

        template<typename T_Particles, typename T_Queue, typename T_Receiver>
        class ReceiveChunksOperation
        {
            struct ReceiveReceiver
            {
                template<typename... T>
                void set_value(T&&...) noexcept
                {
                    owner->afterReceive();
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->fail(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->stop();
                }

                ReceiveChunksOperation* owner;
            };

            struct InsertReceiver
            {
                template<typename... T>
                void set_value(T&&...) noexcept
                {
                    owner->afterInsert();
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->fail(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->stop();
                }

                ReceiveChunksOperation* owner;
            };

            using ReceiveSender = decltype(std::declval<T_Particles&>().getParticlesBuffer().receiveParticles(
                std::declval<T_Queue&>(),
                std::declval<uint32_t>()));
            using InsertSender = decltype(std::declval<T_Particles&>().insertParticlesAsync(
                std::declval<T_Queue&>(),
                std::declval<uint32_t>(),
                std::declval<size_t>()));

            class ReceiveOperation
            {
            public:
                ReceiveOperation(ReceiveSender sender, ReceiveChunksOperation* owner)
                    : operation(std::move(sender).connect(ReceiveReceiver{owner}))
                {
                }

                void start() noexcept
                {
                    operation.start();
                }

            private:
                decltype(std::declval<ReceiveSender&&>().connect(std::declval<ReceiveReceiver>())) operation;
            };

            class InsertOperation
            {
            public:
                InsertOperation(InsertSender sender, ReceiveChunksOperation* owner)
                    : operation(std::move(sender).connect(InsertReceiver{owner}))
                {
                }

                void start() noexcept
                {
                    operation.start();
                }

            private:
                decltype(std::declval<InsertSender&&>().connect(std::declval<InsertReceiver>())) operation;
            };

        public:
            ReceiveChunksOperation(T_Queue& queue, T_Particles& particles, uint32_t exchange, T_Receiver receiver)
                : queue(queue)
                , particles(particles)
                , exchange(exchange)
                , receiver(std::move(receiver))
            {
            }

            ReceiveChunksOperation(ReceiveChunksOperation const&) = delete;
            ReceiveChunksOperation& operator=(ReceiveChunksOperation const&) = delete;
            ReceiveChunksOperation(ReceiveChunksOperation&&) = delete;
            ReceiveChunksOperation& operator=(ReceiveChunksOperation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    maxSize = particles.getParticlesBuffer().getReceiveExchangeStack(exchange).getMaxParticlesCount();
                    startReceive();
                }
                catch(...)
                {
                    fail(std::current_exception());
                }
            }

        private:
            void startReceive()
            {
                receives.emplace_back(particles.getParticlesBuffer().receiveParticles(queue, exchange), this);
                receives.back().start();
            }

            void afterReceive() noexcept
            {
                try
                {
                    lastSize = particles.getParticlesBuffer()
                                   .getReceiveExchangeStack(exchange)
                                   .getHostParticlesCurrentSize();
                    PMACC_ASSERT(lastSize <= maxSize);
                    if(lastSize == 0u)
                    {
                        receiver.set_value();
                        return;
                    }
                    inserts.emplace_back(particles.insertParticlesAsync(queue, exchange, lastSize), this);
                    inserts.back().start();
                }
                catch(...)
                {
                    fail(std::current_exception());
                }
            }

            void afterInsert() noexcept
            {
                try
                {
                    if(lastSize == maxSize)
                        startReceive();
                    else
                        receiver.set_value();
                }
                catch(...)
                {
                    fail(std::current_exception());
                }
            }

            void fail(std::exception_ptr error) noexcept
            {
                receiver.set_error(std::move(error));
            }

            void stop() noexcept
            {
                receiver.set_stopped();
            }

            T_Queue& queue;
            T_Particles& particles;
            uint32_t exchange;
            T_Receiver receiver;
            size_t maxSize = 0u;
            size_t lastSize = 0u;
            // ponytail: retain stage states for synchronous completion; reuse slots if P1 shows this allocation
            // matters.
            std::list<ReceiveOperation> receives;
            std::list<InsertOperation> inserts;
        };
    } // namespace detail

    template<typename T_Particles, typename T_Queue>
    class SendChunksSender
    {
    public:
        using completion_signatures = caravan::CompletionSignatures<
            caravan::ValueSignature<>,
            caravan::ErrorSignature<std::exception_ptr>,
            caravan::StoppedSignature>;

        SendChunksSender(T_Queue& queue, T_Particles& particles, uint32_t exchange)
            : queue(&queue)
            , particles(&particles)
            , exchange(exchange)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::SendChunksOperation<T_Particles, T_Queue, std::decay_t<T_Receiver>>{
                *queue,
                *particles,
                exchange,
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Queue* queue;
        T_Particles* particles;
        uint32_t exchange;
    };

    template<typename T_Particles, typename T_Queue>
    auto sendChunks(T_Queue& queue, T_Particles& particles, uint32_t exchange)
    {
        return SendChunksSender<T_Particles, T_Queue>{queue, particles, exchange};
    }

    template<typename T_Particles, typename T_Queue>
    class ReceiveChunksSender
    {
    public:
        using completion_signatures = caravan::CompletionSignatures<
            caravan::ValueSignature<>,
            caravan::ErrorSignature<std::exception_ptr>,
            caravan::StoppedSignature>;

        ReceiveChunksSender(T_Queue& queue, T_Particles& particles, uint32_t exchange)
            : queue(&queue)
            , particles(&particles)
            , exchange(exchange)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ReceiveChunksOperation<T_Particles, T_Queue, std::decay_t<T_Receiver>>{
                *queue,
                *particles,
                exchange,
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Queue* queue;
        T_Particles* particles;
        uint32_t exchange;
    };

    template<typename T_Particles, typename T_Queue>
    auto receiveChunks(T_Queue& queue, T_Particles& particles, uint32_t exchange)
    {
        return ReceiveChunksSender<T_Particles, T_Queue>{queue, particles, exchange};
    }

    /** Eager runtime-sized adapter for all particle exchange directions. */
    template<typename T_Particles, typename T_Queue>
    caravan::Event spawnCommunication(
        async::Context& context,
        T_Queue& queue,
        T_Particles& particles,
        caravan::Event previous = {})
    {
        using HandleGuardRegion = typename T_Particles::HandleGuardRegion;
        using HandleNotExchanged = typename HandleGuardRegion::HandleNotExchanged;
        auto& buffer = particles.getParticlesBuffer();
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
                    caravan::letValue(
                        caravan::asSender(caravan::whenAll(dependencies)),
                        [&queue, &particles, exchange] { return sendChunks(queue, particles, exchange); }));
                buffer.setSendCompletion(exchange, completion);
                sends.push_back(std::move(completion));
            }
            else
                sends.push_back(context.spawn(
                    caravan::letValue(
                        caravan::asSender(previous),
                        [&queue, &particles, exchange]
                        { return HandleNotExchanged{}.handleOutgoingAsync(queue, particles, exchange); })));

            if(buffer.hasReceiveExchange(exchange))
            {
                std::array dependencies{previous, buffer.receiveCompletion(exchange)};
                auto completion = context.spawn(
                    caravan::letValue(
                        caravan::asSender(caravan::whenAll(dependencies)),
                        [&queue, &particles, exchange] { return receiveChunks(queue, particles, exchange); }));
                buffer.setReceiveCompletion(exchange, completion);
                receives.push_back(std::move(completion));
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
