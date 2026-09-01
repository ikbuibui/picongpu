/* Copyright 2013-2024 Heiko Burau, Rene Widera
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

#pragma once

#include "pmacc/Environment.hpp"
#include "pmacc/fields/SimulationFieldHelper.hpp"
#include "pmacc/mappings/kernel/AreaMapping.hpp"
#include "pmacc/mappings/kernel/ExchangeMapping.hpp"
#include "pmacc/particles/memory/boxes/ParticlesBox.hpp"
#include "pmacc/particles/memory/buffers/ParticlesBuffer.hpp"
#include "pmacc/traits/GetNumWorkers.hpp"

#include <utility>

namespace pmacc
{
    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteGuardParticles(uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

        PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<typename T_Queue>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteGuardParticlesAsync(
        T_Queue& queue,
        uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        return PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)
            .sender(queue, particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<uint32_t T_area>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::deleteParticlesInArea()
    {
        auto const mapper = makeAreaMapper<T_area>(this->cellDescription);

        PMACC_LOCKSTEP_KERNEL(KernelDeleteParticles{})
            .config(mapper.getGridDim(), *particlesBuffer)(particlesBuffer->getDeviceParticleBox(), mapper);
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::reset(uint32_t)
    {
        deleteParticlesInArea<CORE + BORDER + GUARD>();
        particlesBuffer->reset();
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::copyGuardToExchange(uint32_t exchangeType)
    {
        if(particlesBuffer->hasSendExchange(exchangeType))
        {
            ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

            particlesBuffer->getSendExchangeStack(exchangeType).setSize(0);

            PMACC_LOCKSTEP_KERNEL(KernelCopyGuardToExchange{})
                .config(mapper.getGridDim(), *particlesBuffer)(
                    particlesBuffer->getDeviceParticleBox(),
                    particlesBuffer->getSendExchangeStack(exchangeType).getDeviceExchangePushDataBox(),
                    mapper);
        }
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<typename T_Queue>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::copyGuardToExchangeAsync(
        T_Queue& queue,
        uint32_t exchangeType)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        auto stack = particlesBuffer->getSendExchangeStack(exchangeType);
        auto reset = stack.resetAsync(queue);
        auto copy = PMACC_LOCKSTEP_KERNEL(KernelCopyGuardToExchange{})
                        .config(mapper.getGridDim(), *particlesBuffer)
                        .sender(
                            queue,
                            particlesBuffer->getDeviceParticleBox(),
                            stack.getDeviceExchangePushDataBox(),
                            mapper);
        return caravan::alpaka::then(
            caravan::alpaka::then(std::move(reset), std::move(copy)),
            stack.publishDeviceSizes(queue));
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    void ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::insertParticles(uint32_t exchangeType)
    {
        if(particlesBuffer->hasReceiveExchange(exchangeType))
        {
            size_t numParticles = 0u;
            if(Environment<>::get().isMpiDirectEnabled())
                numParticles = particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceCurrentSize();
            else
                numParticles = particlesBuffer->getReceiveExchangeStack(exchangeType).getHostCurrentSize();

            if(numParticles != 0u)
            {
                ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);

                PMACC_LOCKSTEP_KERNEL(KernelInsertParticles{})
                    .config(numParticles, *particlesBuffer)(
                        particlesBuffer->getDeviceParticleBox(),
                        particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                        mapper);
            }
        }
    }

    template<typename T_ParticleDescription, class MappingDesc, typename T_DeviceHeap>
    template<typename T_Queue>
    auto ParticlesBase<T_ParticleDescription, MappingDesc, T_DeviceHeap>::insertParticlesAsync(
        T_Queue& queue,
        uint32_t exchangeType,
        size_t numParticles)
    {
        ExchangeMapping<GUARD, MappingDesc> mapper(this->cellDescription, exchangeType);
        return PMACC_LOCKSTEP_KERNEL(KernelInsertParticles{})
            .config(numParticles, *particlesBuffer)
            .sender(
                queue,
                particlesBuffer->getDeviceParticleBox(),
                particlesBuffer->getReceiveExchangeStack(exchangeType).getDeviceExchangePopDataBox(),
                mapper);
    }

} // namespace pmacc

#include "pmacc/particles/AsyncCommunicationImpl.hpp"
