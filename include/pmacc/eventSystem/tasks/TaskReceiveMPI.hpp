/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"

#include <memory>

#include <mpi.h>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskReceiveMPI : public MPITask
    {
    public:
        TaskReceiveMPI(Exchange<TYPE, DIM>* exchange) : MPITask(), exchange(exchange)
        {
        }

        void init() override
        {
            auto cPtr = exchange->getCPtrCapacity();
            auto& communicator = Environment<DIM>::get().EnvironmentController().getCommunicator();
            if(communicator.usesMpiContext())
            {
                future = communicator.startReceiveAsync(
                    exchange->getExchangeType(),
                    cPtr.asCharPtr(),
                    cPtr.sizeInBytes(),
                    exchange->getCommunicationTag());
            }
            else
            {
                request = communicator.startReceive(
                    exchange->getExchangeType(),
                    cPtr.asCharPtr(),
                    cPtr.sizeInBytes(),
                    exchange->getCommunicationTag());
            }
        }

        bool executeIntern() override
        {
            if(this->isFinished())
                return true;

            if(future.valid())
            {
                Environment<DIM>::get().EnvironmentController().getCommunicator().progressAsync();
                if(future.state() == caravan::CompletionState::pending)
                    return false;
                receivedBytes = static_cast<int>(future.result().bytes);
                setFinished();
                return true;
            }

            if(request == nullptr)
                throw std::runtime_error("request was nullptr (call executeIntern after freed");

            int flag = 0;
            MPI_CHECK(MPI_Test(request, &flag, &status));

            if(flag) // finished
            {
                delete request;
                request = nullptr;
                setFinished();
                return true;
            }
            return false;
        }

        ~TaskReceiveMPI() override
        {
            //! \todo this make problems because we send bytes and not combined types
            if(!future.valid())
                MPI_CHECK_NO_EXCEPT(MPI_Get_count(&status, MPI_CHAR, &receivedBytes));

            std::unique_ptr<IEventData> edata = std::make_unique<EventDataReceive>(nullptr, receivedBytes);

            notify(this->myId, RECVFINISHED, edata.get()); /*add notify her*/
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return std::string("TaskReceiveMPI exchange type=") + std::to_string(exchange->getExchangeType());
        }

    private:
        Exchange<TYPE, DIM>* exchange;
        caravan::Future<caravan::ReceiveResult> future;
        MPI_Request* request{nullptr};
        MPI_Status status;
        int receivedBytes{0};
    };

} // namespace pmacc
