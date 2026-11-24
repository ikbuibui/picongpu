/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Benjamin Worpitz
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
#include "pmacc/assert.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/events/EventTask.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/boxes/ExchangePopDataBox.hpp"
#include "pmacc/particles/memory/boxes/ExchangePushDataBox.hpp"
#include "pmacc/particles/memory/buffers/ExchangeCounters.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace pmacc
{
    /** Counter slots of one send direction.
     *
     * The payload counter tracks the number of pushed border frames, the index-entry counter the
     * number of pushed per-supercell exchange index entries. Both share one allocation so a reset
     * and a publish stay a single operation.
     */
    enum class ExchangeCounter : uint32_t
    {
        payload = 0u,
        indexEntries = 1u
    };

    //! number of counters shared by one send direction
    inline constexpr uint32_t numExchangeCounters = 2u;

    //! counter storage of one send direction
    using ExchangeCounterBlock = ExchangeCounters<vint_t, numExchangeCounters>;

    namespace detail
    {
        //! index of a counter slot inside ExchangeCounterBlock
        constexpr uint32_t counterIndex(ExchangeCounter const slot)
        {
            return static_cast<uint32_t>(slot);
        }

        //! logical size of the buffer the MPI transfers work on
        template<typename T_Exchange>
        size_t transportSize(T_Exchange& exchange)
        {
            if(::pmacc::Environment<>::get().isMpiDirectEnabled())
                return exchange.getDeviceBuffer().size();
            return exchange.getHostBuffer().size();
        }

        //! capacity of the buffer the MPI transfers work on
        template<typename T_Exchange>
        size_t transportCapacity(T_Exchange& exchange)
        {
            if(::pmacc::Environment<>::get().isMpiDirectEnabled())
                return exchange.getDeviceBuffer().capacityND().productOfComponents();
            return exchange.getHostBuffer().capacityND().productOfComponents();
        }
    } // namespace detail

    /** Send view of a (border frame, exchange index) pair.
     *
     * Borrows the two exchanges and the persistent counter block of one send direction; the
     * owning ParticlesBuffer must outlive this view.
     *
     * The counter block is the only carrier of the pack result: reset() hands it to the packing
     * kernel and publish() copies it back and stamps the transport lengths of both exchange
     * buffers, so the generic transfer tasks keep reading the buffer's logical size.
     */
    template<class FRAME, class FRAMEINDEX, unsigned DIM>
    class SendExchangeStack
    {
        static_assert(
            std::is_same_v<typename ExchangeCounterBlock::CountType, vint_t>,
            "the exchange counters must use the same type as the exchange index entry arithmetic");

    public:
        SendExchangeStack(
            Exchange<FRAME, DIM1>& stack,
            Exchange<FRAMEINDEX, DIM1>& stackIndexer,
            ExchangeCounterBlock& counters)
            : stack(stack)
            , stackIndexer(stackIndexer)
            , counters(counters)
        {
        }

        /**
         * Returns a PushDataBox for the internal DeviceBuffer.
         *
         * @return PushDataBox for device buffer
         */
        ExchangePushDataBox<vint_t, FRAME, DIM> getDeviceExchangePushDataBox()
        {
            auto const maxSize = stack.getDeviceBuffer().capacityND().productOfComponents();
            PMACC_ASSERT(maxSize <= static_cast<size_t>(std::numeric_limits<vint_t>::max()));
            return ExchangePushDataBox<vint_t, FRAME, DIM>(
                stack.getDeviceBuffer().data(),
                counters.deviceData() + detail::counterIndex(ExchangeCounter::payload),
                static_cast<vint_t>(maxSize),
                PushDataBox<vint_t, FRAMEINDEX>(
                    stackIndexer.getDeviceBuffer().data(),
                    counters.deviceData() + detail::counterIndex(ExchangeCounter::indexEntries)));
        }

        /** Zero both device counters; the returned task is ordered before the packing kernel. */
        EventTask reset()
        {
            return Environment<>::get().Factory().createTaskResetExchangeCounters(counters);
        }

        /** Queue the device-to-host copy of both counters and return its task.
         *
         * Non-blocking: the caller ends the transaction it created this task in, polls the
         * returned event, and only then reads payloadCount() or calls applyPublishedLengths().
         */
        EventTask startPublish()
        {
            return Environment<>::get().Factory().createTaskPublishExchangeCounters(counters);
        }

        /** Stamp the published counters as the logical sizes of the transport buffers.
         *
         * Host-side only; call it after the publish task finished and before the transfer reads
         * the buffer sizes.
         */
        void applyPublishedLengths()
        {
            stack.getDeviceBuffer().setSize(payloadCount());
            stackIndexer.getDeviceBuffer().setSize(indexCount());
        }

        //! number of border frames produced by the last completed pack kernel
        size_t payloadCount() const
        {
            return counters.hostValue(detail::counterIndex(ExchangeCounter::payload));
        }

        //! number of exchange index entries produced by the last completed pack kernel
        size_t indexCount() const
        {
            return counters.hostValue(detail::counterIndex(ExchangeCounter::indexEntries));
        }

        //! maximum number of border frames one chunk can carry
        size_t maxPayloadCount()
        {
            return detail::transportCapacity(stack);
        }

    private:
        Exchange<FRAME, DIM1>& stack;
        Exchange<FRAMEINDEX, DIM1>& stackIndexer;
        ExchangeCounterBlock& counters;
    };

    /** Receive view of a (border frame, exchange index) pair.
     *
     * Borrows the two exchanges; the owning ParticlesBuffer must outlive this view. The received
     * lengths live in the exchanges' logical sizes, which the receive tasks set from the MPI
     * metadata.
     */
    template<class FRAME, class FRAMEINDEX, unsigned DIM>
    class ReceiveExchangeStack
    {
    public:
        ReceiveExchangeStack(Exchange<FRAME, DIM1>& stack, Exchange<FRAMEINDEX, DIM1>& stackIndexer)
            : stack(stack)
            , stackIndexer(stackIndexer)
        {
        }

        /**
         * Returns a PopDataBox for the internal DeviceBuffer.
         *
         * @return PopDataBox for device buffer
         */
        ExchangePopDataBox<vint_t, FRAME, DIM> getDeviceExchangePopDataBox()
        {
            return ExchangePopDataBox<vint_t, FRAME, DIM>(
                stack.getDeviceBuffer().getDataBox(),
                stackIndexer.getDeviceBuffer().getDataBox());
        }

        //! number of received border frames
        size_t payloadCount()
        {
            return detail::transportSize(stack);
        }

        //! number of received exchange index entries
        size_t indexCount()
        {
            return detail::transportSize(stackIndexer);
        }

        //! maximum number of border frames one chunk can carry
        size_t maxPayloadCount()
        {
            return detail::transportCapacity(stack);
        }

    private:
        Exchange<FRAME, DIM1>& stack;
        Exchange<FRAMEINDEX, DIM1>& stackIndexer;
    };
} // namespace pmacc
