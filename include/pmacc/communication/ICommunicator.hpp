/* Copyright 2013-2024 Rene Widera, Wolfgang Hoenig, Benjamin Worpitz
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
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

#include <caravan/mpi.hpp>

namespace pmacc
{
    /*! Interface for communication
     */
    class ICommunicator
    {
    public:
        /*! returns available communication partners
         *
         * returns a mask with neighbors, e.g. if there is a right neighbor result.isSet(RIGHT) returns true
         */
        virtual Mask const& getCommunicationMask() const = 0;

        /*! moves all GPUs from top to bottom (y-coordinate)
         *
         * @return true if the position of gpu is switched to the end, else false
         */
        virtual bool slide() = 0;

        /*! slides multiple times
         *
         * @param[in] numSlides number of slides
         * @return true if the position of gpu is switched to the end, else false
         */
        virtual bool setStateAfterSlides(size_t numSlides) = 0;

        virtual caravan::mpi::OperationSender<caravan::SendResult> send(
            uint32_t ex,
            char const* sendData,
            size_t sendBytes,
            uint32_t tag)
            = 0;

        virtual caravan::mpi::OperationSender<caravan::ReceiveResult> receive(
            uint32_t ex,
            char* receiveData,
            size_t receiveBytes,
            uint32_t tag)
            = 0;

        virtual caravan::Future<caravan::SendResult> startSendAsync(
            uint32_t ex,
            char const* sendData,
            size_t sendBytes,
            uint32_t tag)
            = 0;

        virtual caravan::Future<caravan::ReceiveResult> startReceiveAsync(
            uint32_t ex,
            char* receiveData,
            size_t receiveBytes,
            uint32_t tag)
            = 0;

        /** Execute ready PMacc-side completions from asynchronous backends. */
        virtual void progressAsync() = 0;

        virtual int getRank() = 0;

        /*! Return which of the three directions are periodic
         *
         * @return for each direction a false (0) or true(1) value
         */
        virtual DataSpace<DIM3> getPeriodic() const = 0;
    };

} // namespace pmacc
