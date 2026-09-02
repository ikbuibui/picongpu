/* Copyright 2016-2024 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include "IdProvider.hpp"
#include "memory/SuperCell.hpp"

#include <pmacc/HandleGuardRegion.hpp>
#include <pmacc/fields/Communication.hpp>
#include <pmacc/particles/Communication.hpp>
#include <pmacc/particles/policies/DoNothing.hpp>

#include <exception>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

namespace
{
    enum class FailurePoint
    {
        none,
        packingCompletion,
        packingStopped,
        sizeExtraction,
        sendInitiation,
        sendCompletion,
        receiveInitiation,
        receiveCompletion,
        insertion,
        retrySetup
    };

    caravan::Event failedEvent()
    {
        caravan::EventSource source;
        source.setFailed(std::make_exception_ptr(std::runtime_error("injected particle failure")));
        return source.event();
    }

    caravan::Event stoppedEvent()
    {
        caravan::EventSource source;
        source.setStopped();
        return source.event();
    }

    struct MockStack
    {
        size_t getMaxParticlesCount() const
        {
            return 2u;
        }

        size_t getDeviceParticlesCurrentSize() const
        {
            if(*failure == FailurePoint::sizeExtraction)
                throw std::runtime_error("injected particle size failure");
            return *size;
        }

        size_t getHostParticlesCurrentSize() const
        {
            if(*failure == FailurePoint::sizeExtraction)
                throw std::runtime_error("injected particle size failure");
            return *size;
        }

        size_t const* size;
        FailurePoint const* failure;
    };

    struct MockParticlesBuffer
    {
        bool hasSendExchange(uint32_t exchange) const
        {
            return exchange == 1u;
        }

        bool hasReceiveExchange(uint32_t exchange) const
        {
            return exchange == 1u;
        }

        MockStack getSendExchangeStack(uint32_t) const
        {
            return {&sendSize, &failure};
        }

        MockStack getReceiveExchangeStack(uint32_t) const
        {
            return {&receiveSize, &failure};
        }

        caravan::Event sendCompletion(uint32_t) const
        {
            return {};
        }

        caravan::Event receiveCompletion(uint32_t) const
        {
            return {};
        }

        void setSendCompletion(uint32_t, caravan::Event)
        {
        }

        void setReceiveCompletion(uint32_t, caravan::Event)
        {
        }

        template<typename T_Queue>
        auto sendParticles(T_Queue& queue, uint32_t)
        {
            return caravan::alpaka::submit(
                queue,
                [this](T_Queue&)
                {
                    if(failure == FailurePoint::sendInitiation)
                        throw std::runtime_error("injected particle send initiation failure");
                    if(failure == FailurePoint::sendCompletion)
                        throw std::runtime_error("injected particle send completion failure");
                });
        }

        template<typename T_Queue>
        auto receiveParticles(T_Queue& queue, uint32_t)
        {
            return caravan::alpaka::submit(
                queue,
                [this](T_Queue&)
                {
                    if(failure == FailurePoint::receiveInitiation)
                        throw std::runtime_error("injected particle receive initiation failure");
                    receiveSize = receiveChunks.at(receiveChunk++);
                    if(failure == FailurePoint::receiveCompletion)
                        throw std::runtime_error("injected particle receive completion failure");
                });
        }

        std::array<size_t, 2u> sendChunks{2u, 1u};
        std::array<size_t, 2u> receiveChunks{2u, 1u};
        size_t sendChunk = 0u;
        size_t receiveChunk = 0u;
        size_t sendSize = 0u;
        size_t receiveSize = 0u;
        FailurePoint failure = FailurePoint::none;
    };

    struct MockParticles
    {
        using HandleGuardRegion
            = pmacc::HandleGuardRegion<pmacc::particles::policies::DoNothing, pmacc::particles::policies::DoNothing>;

        struct FrameType
        {
            static char const* getName()
            {
                return "mock";
            }
        };

        static constexpr uint32_t dim = TEST_DIM;

        MockParticlesBuffer& getParticlesBuffer()
        {
            return buffer;
        }

        template<typename T_Queue>
        auto copyGuardToExchangeAsync(T_Queue&, uint32_t)
        {
            if(buffer.failure == FailurePoint::retrySetup && buffer.sendChunk != 0u)
                throw std::runtime_error("injected particle retry failure");
            if(buffer.failure == FailurePoint::packingCompletion)
                return caravan::asSender(failedEvent());
            if(buffer.failure == FailurePoint::packingStopped)
                return caravan::asSender(stoppedEvent());
            buffer.sendSize = buffer.sendChunks.at(buffer.sendChunk++);
            return caravan::asSender(caravan::readyEvent());
        }

        template<typename T_Queue>
        auto insertParticlesAsync(T_Queue&, uint32_t, size_t count)
        {
            if(buffer.failure == FailurePoint::insertion)
                return caravan::asSender(failedEvent());
            inserted += count;
            return caravan::asSender(caravan::readyEvent());
        }

        template<typename T_Queue>
        auto fillBorderGapsAsync(T_Queue& queue)
        {
            return caravan::alpaka::submit(queue, [this](T_Queue&) { gapsFilled = true; });
        }

        MockParticlesBuffer buffer;
        size_t inserted = 0u;
        bool gapsFilled = false;
    };
} // namespace

TEST_CASE("Particle chunk senders are lazy", "[particles][async]")
{
    auto& queue = pmacc::Environment<>::get().QueueController().getNextStream()->borrowAlpakaQueue();
    pmacc::async::Context context;
    MockParticles particles;
    auto sender = pmacc::particles::sendChunks(queue, particles, 1u);
    static_assert(caravan::Sender<decltype(sender)>);
    static_assert(caravan::Sender<decltype(pmacc::particles::receiveChunks(queue, particles, 1u))>);
    CHECK(particles.buffer.sendChunk == 0u);
    context.wait(context.spawn(std::move(sender)));
    CHECK(particles.buffer.sendChunk == 2u);
}

TEST_CASE("Particle chunk senders propagate stopped completion", "[particles][async]")
{
    auto& queue = pmacc::Environment<>::get().QueueController().getNextStream()->borrowAlpakaQueue();
    pmacc::async::Context context;
    MockParticles particles;
    particles.buffer.failure = FailurePoint::packingStopped;
    CHECK_THROWS_AS(
        context.wait(context.spawn(pmacc::particles::sendChunks(queue, particles, 1u))),
        caravan::StoppedError);
}

TEST_CASE("Particle communication handles exact and partial chunks", "[particles][async]")
{
    auto& queue = pmacc::Environment<>::get().QueueController().getNextStream()->borrowAlpakaQueue();
    pmacc::async::Context context;
    MockParticles particles;
    context.wait(pmacc::particles::spawnCommunication(context, queue, particles));
    CHECK(particles.buffer.sendChunk == 2u);
    CHECK(particles.buffer.receiveChunk == 2u);
    CHECK(particles.inserted == 3u);
    CHECK(particles.gapsFilled);
}

TEST_CASE("Particle communication handles empty chunks", "[particles][async]")
{
    auto& queue = pmacc::Environment<>::get().QueueController().getNextStream()->borrowAlpakaQueue();
    pmacc::async::Context context;
    MockParticles particles;
    particles.buffer.sendChunks = {0u, 0u};
    particles.buffer.receiveChunks = {0u, 0u};
    context.wait(pmacc::particles::spawnCommunication(context, queue, particles));
    CHECK(particles.buffer.sendChunk == 1u);
    CHECK(particles.buffer.receiveChunk == 1u);
    CHECK(particles.inserted == 0u);
    CHECK(particles.gapsFilled);
}

TEST_CASE("Particle communication forwards callback failures", "[particles][async]")
{
    auto& queue = pmacc::Environment<>::get().QueueController().getNextStream()->borrowAlpakaQueue();
    for(auto const failure :
        {FailurePoint::packingCompletion,
         FailurePoint::sizeExtraction,
         FailurePoint::sendInitiation,
         FailurePoint::sendCompletion,
         FailurePoint::receiveInitiation,
         FailurePoint::receiveCompletion,
         FailurePoint::insertion,
         FailurePoint::retrySetup})
    {
        pmacc::async::Context context;
        MockParticles particles;
        particles.buffer.failure = failure;
        CHECK_THROWS_AS(
            context.wait(pmacc::particles::spawnCommunication(context, queue, particles)),
            std::runtime_error);
    }
}
