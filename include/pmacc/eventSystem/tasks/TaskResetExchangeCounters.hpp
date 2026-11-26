/* Copyright 2026 PIConGPU contributors
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

#include "pmacc/eventSystem/events/kernelEvents.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>

namespace pmacc
{
    /** Zero the device counters of one exchange counter group.
     *
     * The fill is queued on the device queue, so it is ordered before the packing kernel that
     * follows in the same transaction. A device fill replaces the former one-thread kernel that
     * wrote a buffer's current size into device memory.
     *
     * @tparam T_Counters exchange counter group, see pmacc::ExchangeCounters
     */
    template<typename T_Counters>
    class TaskResetExchangeCounters : public DeviceTask
    {
    public:
        TaskResetExchangeCounters(T_Counters& counters) : DeviceTask(), counters(&counters)
        {
        }

        ~TaskResetExchangeCounters() override
        {
            notify(this->myId, SETVALUE, nullptr);
        }

        void init() override
        {
            auto queue = this->getAlpakaQueue();
            alpaka::memset(queue, counters->deviceBuffer(), std::uint8_t{0});
            activate();
        }

        bool executeIntern() override
        {
            return isFinished();
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return "TaskResetExchangeCounters";
        }

    private:
        T_Counters* counters;
    };
} // namespace pmacc
