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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/events/kernelEvents.hpp"
#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

#include <alpaka/alpaka.hpp>

namespace pmacc
{
    /** Copy all counters of one exchange counter group from the device to the host.
     *
     * All counters of a group share one allocation, so a whole group is published with a single
     * copy instead of one copy per counter. The host values are only valid once this task is
     * finished; use an observer or EventTask::waitForFinished() before reading them.
     *
     * @tparam T_Counters exchange counter group, see pmacc::ExchangeCounters
     */
    template<typename T_Counters>
    class TaskPublishExchangeCounters : public DeviceTask
    {
    public:
        TaskPublishExchangeCounters(T_Counters& counters) : DeviceTask(), counters(&counters)
        {
        }

        ~TaskPublishExchangeCounters() override
        {
            notify(this->myId, GETVALUE, nullptr);
        }

        void init() override
        {
            auto queue = this->getAlpakaQueue();
            alpaka::memcpy(queue, counters->hostBuffer(), counters->deviceBuffer());
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
            return "TaskPublishExchangeCounters";
        }

    private:
        T_Counters* counters;
    };
} // namespace pmacc
