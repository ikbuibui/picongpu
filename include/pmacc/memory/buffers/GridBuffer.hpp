/* Copyright 2013-2024 Rene Widera, Benjamin Worpitz, Alexander Grund
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
#include "pmacc/dimensions/GridLayout.hpp"
#include "pmacc/memory/buffers/Exchange.hpp"
#include "pmacc/memory/buffers/HostDeviceBuffer.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace pmacc
{
    namespace privateGridBuffer
    {
        class UniquTag
        {
        public:
            static UniquTag& getInstance()
            {
                static UniquTag instance;
                return instance;
            }

            bool isTagUniqu(uint32_t tag)
            {
                bool isUniqu = tags.find(tag) == tags.end();
                if(isUniqu)
                    tags.insert(tag);
                return isUniqu;
            }

        private:
            UniquTag() = default;

            /**
             * Constructor
             */
            UniquTag(UniquTag const&)
            {
            }

            std::set<uint32_t> tags;
        };

    } // end namespace privateGridBuffer

    /**
     * GridBuffer represents a DIM-dimensional buffer which exists on the host as well as on the device.
     *
     * GridBuffer combines a HostBuffer and a DeviceBuffer with equal sizes.
     * Additionally, it allows sending data from and receiving data to these buffers.
     * Buffers consist of core data which may be surrounded by border data.
     *
     * @tparam TYPE datatype for internal Host- and DeviceBuffer
     * @tparam DIM dimension of the buffers
     * @tparam BORDERTYPE optional type for border data in the buffers. TYPE is used by default.
     */
    template<class TYPE, unsigned DIM, class BORDERTYPE = TYPE>
    class GridBuffer : public HostDeviceBuffer<TYPE, DIM>
    {
        using Parent = HostDeviceBuffer<TYPE, DIM>;

    public:
        using DataBoxType = typename Parent::DataBoxType;

        /**
         * Constructor.
         *
         * @param gridLayout layout of the buffers, including border-cells
         * @param sizeOnDevice if true, size information exists on device, too.
         */
        GridBuffer(GridLayout<DIM> const& gridLayout, bool sizeOnDevice = false)
            : Parent(gridLayout.sizeND(), sizeOnDevice)
            , hasOneExchange(false)
            , gridLayout(gridLayout)
            , maxExchange(0)
        {
        }

        /**
         * Constructor.
         *
         * @param dataSpace DataSpace representing buffer size without border-cells
         * @param sizeOnDevice if true, internal buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         */
        GridBuffer(DataSpace<DIM> const& dataSpace, bool sizeOnDevice = false)
            : Parent(dataSpace, sizeOnDevice)
            , hasOneExchange(false)
            , gridLayout(dataSpace)
            , maxExchange(0)
        {
        }

        /**
         * Constructor.
         *
         * @param otherDeviceBuffer DeviceBuffer which should be used instead of creating own DeviceBuffer
         * @param gridLayout layout of the buffers, including border-cells
         * @param sizeOnDevice if true, internal buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         */
        GridBuffer(
            DeviceBuffer<TYPE, DIM>& otherDeviceBuffer,
            GridLayout<DIM> const& gridLayout,
            bool sizeOnDevice = false)
            : Parent(otherDeviceBuffer, gridLayout.sizeND(), sizeOnDevice)
            , hasOneExchange(false)
            , gridLayout(gridLayout)
            , maxExchange(0)
        {
        }

        GridBuffer(
            HostBuffer<TYPE, DIM>& otherHostBuffer,
            DataSpace<DIM> const& offsetHost,
            DeviceBuffer<TYPE, DIM>& otherDeviceBuffer,
            DataSpace<DIM> const& offsetDevice,
            GridLayout<DIM> const& gridLayout,
            bool sizeOnDevice = false)
            : Parent(otherHostBuffer, offsetHost, otherDeviceBuffer, offsetDevice, gridLayout.sizeND(), sizeOnDevice)
            , hasOneExchange(false)
            , gridLayout(gridLayout)
            , maxExchange(0)
        {
        }

        /**
         * Add Exchange in GridBuffer memory space.
         *
         * An Exchange is added to this GridBuffer. The exchange buffers use
         * the same memory as this GridBuffer.
         *
         * @param dataPlace place where received data is stored [GUARD | BORDER]
         *        if dataPlace=GUARD than copy other BORDER to my GUARD
         *        if dataPlace=BORDER than copy other GUARD to my BORDER
         * @param receive a Mask which describes the directions for the exchange
         * @param guardingCells number of guarding cells in each dimension
         * @param communicationTag unique tag/id for communication
         *        has to be the same when this method is called multiple times for the same object
         *        (with non-overlapping masks)
         * @param sizeOnDeviceSend if true, internal send buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         * @param sizeOnDeviceReceive if true, internal receive buffers must store their
         *        size additionally on the device
         */
        void addExchange(
            uint32_t dataPlace,
            Mask const& receive,
            DataSpace<DIM> guardingCells,
            uint32_t communicationTag,
            bool sizeOnDeviceSend,
            bool sizeOnDeviceReceive)
        {
            if(hasOneExchange && (communicationTag != lastUsedCommunicationTag))
                throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");

            lastUsedCommunicationTag = communicationTag;

            receiveMask = receiveMask + receive;
            sendMask = this->receiveMask.getMirroredMask();
            Mask send = receive.getMirroredMask();


            for(uint32_t ex = 1; ex < -12 * (int) DIM + 6 * (int) DIM * (int) DIM + 9; ++ex)
            {
                if(send.isSet(ex))
                {
                    /* This operation relies on communicationTag being relatively small, so that the resulting
                     * uniqCommunicationTag fits the range of valid tags
                     */
                    uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;

                    if(!hasOneExchange && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                    {
                        std::stringstream message;
                        message << "unique exchange communication tag (" << uniqCommunicationTag
                                << ") which is created from communicationTag (" << communicationTag
                                << ") already used for other GridBuffer exchange";
                        throw std::runtime_error(message.str());
                    }
                    hasOneExchange = true;

                    if(sendExchanges[ex] != nullptr)
                    {
                        throw std::runtime_error("Exchange already added!");
                    }

                    maxExchange = std::max(maxExchange, ex + 1u);
                    sendExchanges[ex] = std::make_unique<Exchange<BORDERTYPE, DIM>>(
                        this->getDeviceBuffer(),
                        gridLayout,
                        guardingCells,
                        (ExchangeType) ex,
                        uniqCommunicationTag,
                        dataPlace == GUARD ? BORDER : GUARD,
                        sizeOnDeviceSend);
                    ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                    maxExchange = std::max(maxExchange, recvex + 1u);
                    receiveExchanges[recvex] = std::make_unique<Exchange<BORDERTYPE, DIM>>(
                        this->getDeviceBuffer(),
                        gridLayout,
                        guardingCells,
                        recvex,
                        uniqCommunicationTag,
                        dataPlace == GUARD ? GUARD : BORDER,
                        sizeOnDeviceReceive);
                }
            }
        }

        /**
         * Add Exchange in GridBuffer memory space.
         *
         * An Exchange is added to this GridBuffer. The exchange buffers use
         * the same memory as this GridBuffer.
         *
         * @param dataPlace place where received data is stored [GUARD | BORDER]
         *        if dataPlace=GUARD than copy other BORDER to my GUARD
         *        if dataPlace=BORDER than copy other GUARD to my BORDER
         * @param receive a Mask which describes the directions for the exchange
         * @param guardingCells number of guarding cells in each dimension
         * @param communicationTag unique tag/id for communication
         * @param sizeOnDevice if true, internal buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         */
        void addExchange(
            uint32_t dataPlace,
            Mask const& receive,
            DataSpace<DIM> guardingCells,
            uint32_t communicationTag,
            bool sizeOnDevice = false)
        {
            addExchange(dataPlace, receive, guardingCells, communicationTag, sizeOnDevice, sizeOnDevice);
        }

        /**
         * Add Exchange in dedicated memory space.
         *
         * An Exchange is added to this GridBuffer. The exchange buffers use
         * the their own memory instead of using the GridBuffer's memory space.
         *
         * @param receive a Mask which describes the directions for the exchange
         * @param dataSpace size of the newly created exchange buffer in each dimension
         * @param communicationTag unique tag/id for communication
         * @param sizeOnDeviceSend if true, internal send buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         * @param sizeOnDeviceReceive if true, internal receive buffers must store their
         *        size additionally on the device
         */
        void addExchangeBuffer(
            Mask const& receive,
            DataSpace<DIM> const& dataSpace,
            uint32_t communicationTag,
            bool sizeOnDeviceSend,
            bool sizeOnDeviceReceive)
        {
            if(hasOneExchange && (communicationTag != lastUsedCommunicationTag))
                throw std::runtime_error("It is not allowed to give the same GridBuffer different communicationTags");
            lastUsedCommunicationTag = communicationTag;


            /*don't create buffer with 0 (zero) elements*/
            if(dataSpace.productOfComponents() != 0)
            {
                receiveMask = receiveMask + receive;
                sendMask = this->receiveMask.getMirroredMask();
                Mask send = receive.getMirroredMask();
                for(uint32_t ex = 1; ex < 27; ++ex)
                {
                    if(send.isSet(ex))
                    {
                        /* This operation relies on communicationTag being relatively small, so that the resulting
                         * uniqCommunicationTag fits the range of valid tags
                         */
                        uint32_t uniqCommunicationTag = (communicationTag << 5) | ex;
                        if(!hasOneExchange
                           && !privateGridBuffer::UniquTag::getInstance().isTagUniqu(uniqCommunicationTag))
                        {
                            std::stringstream message;
                            message << "unique exchange communication tag (" << uniqCommunicationTag
                                    << ") which is created from communicationTag (" << communicationTag
                                    << ") already used for other GridBuffer exchange";
                            throw std::runtime_error(message.str());
                        }
                        hasOneExchange = true;

                        if(sendExchanges[ex] != nullptr)
                        {
                            throw std::runtime_error("Exchange already added!");
                        }

                        // GridLayout<DIM> memoryLayout(size);
                        maxExchange = std::max(maxExchange, ex + 1u);
                        sendExchanges[ex] = std::make_unique<Exchange<BORDERTYPE, DIM>>(
                            /*memoryLayout*/ dataSpace,
                            ex,
                            uniqCommunicationTag,
                            sizeOnDeviceSend);

                        ExchangeType recvex = Mask::getMirroredExchangeType(ex);
                        maxExchange = std::max(maxExchange, recvex + 1u);
                        receiveExchanges[recvex] = std::make_unique<Exchange<BORDERTYPE, DIM>>(
                            /*memoryLayout*/ dataSpace,
                            recvex,
                            uniqCommunicationTag,
                            sizeOnDeviceReceive);
                    }
                }
            }
        }

        /**
         * Add Exchange in dedicated memory space.
         *
         * An Exchange is added to this GridBuffer. The exchange buffers use
         * the their own memory instead of using the GridBuffer's memory space.
         *
         * @param receive a Mask which describes the directions for the exchange
         * @param dataSpace size of the newly created exchange buffer in each dimension
         * @param communicationTag unique tag/id for communication
         * @param sizeOnDevice if true, internal buffers must store their
         *        size additionally on the device
         *        (as we keep this information coherent with the host, it influences
         *        performance on host-device copies, but some algorithms on the device
         *        might need to know the size of the buffer)
         */
        void addExchangeBuffer(
            Mask const& receive,
            DataSpace<DIM> const& dataSpace,
            uint32_t communicationTag,
            bool sizeOnDevice = false)
        {
            addExchangeBuffer(receive, dataSpace, communicationTag, sizeOnDevice, sizeOnDevice);
        }

        /**
         * Returns whether this GridBuffer has an Exchange for sending in ex direction.
         *
         * @param ex exchange direction to query
         * @return true if send exchanges with ex direction exist, otherwise false
         */
        bool hasSendExchange(uint32_t ex) const
        {
            return ((sendExchanges[ex] != nullptr) && (getSendMask().isSet(ex)));
        }

        /**
         * Returns whether this GridBuffer has an Exchange for receiving from ex direction.
         *
         * @param ex exchange direction to query
         * @return true if receive exchanges with ex direction exist, otherwise false
         */
        bool hasReceiveExchange(uint32_t ex) const
        {
            return ((receiveExchanges[ex] != nullptr) && (getReceiveMask().isSet(ex)));
        }

        /**
         * Returns the Exchange for sending data in ex direction.
         *
         * Returns an Exchange which for sending data from
         * this GridBuffer in the direction described by ex.
         *
         * @param ex the direction to query
         * @return the Exchange for sending data
         */
        Exchange<BORDERTYPE, DIM>& getSendExchange(uint32_t ex) const
        {
            return *sendExchanges[ex];
        }

        /**
         * Returns the Exchange for receiving data from ex direction.
         *
         * Returns an Exchange which for receiving data to
         * this GridBuffer from the direction described by ex.
         *
         * @param ex the direction to query
         * @return the Exchange for receiving data
         */
        Exchange<BORDERTYPE, DIM>& getReceiveExchange(uint32_t ex) const
        {
            return *receiveExchanges[ex];
        }

        /**
         * Returns the Mask describing send exchanges
         *
         * @return Mask for send exchanges
         */
        Mask getSendMask() const
        {
            return (Environment<DIM>::get().GridController().getCommunicationMask() & sendMask);
        }

        /**
         * Returns the Mask describing receive exchanges
         *
         * @return Mask for receive exchanges
         */
        Mask getReceiveMask() const
        {
            return (Environment<DIM>::get().GridController().getCommunicationMask() & receiveMask);
        }

        /**
         * Starts sync data from own device buffer to neighbor device buffer.
         *
         * Asynchronously starts synchronization data from internal DeviceBuffer using added
         * Exchange buffers.
         * This operation runs sequential to other code but intern asynchronous
         *
         */
        caravan::Event sendCompletion(uint32_t exchange) const
        {
            return sendCompletions[exchange];
        }

        caravan::Event receiveCompletion(uint32_t exchange) const
        {
            return receiveCompletions[exchange];
        }

        void setSendCompletion(uint32_t exchange, caravan::Event completion)
        {
            sendCompletions[exchange] = std::move(completion);
        }

        void setReceiveCompletion(uint32_t exchange, caravan::Event completion)
        {
            receiveCompletions[exchange] = std::move(completion);
        }

        /** Describe one lazy send for an active exchange direction. */
        auto send(uint32_t exchange)
        {
            return sendExchanges[exchange]->send();
        }

        /** Describe one lazy receive for an active exchange direction. */
        auto receive(uint32_t exchange)
        {
            return receiveExchanges[exchange]->receive();
        }

        /** Describe a receive which publishes queue-side readiness before device quiescence. */
        auto receiveSubmitted(uint32_t exchange)
        {
            return receiveExchanges[exchange]->receiveSubmitted();
        }

        /** Describe all dynamically selected exchange directions as one lazy aggregate.
         *
         * Direction retirement events still serialize staging-buffer reuse, but branch completion is joined
         * directly instead of being transferred through ControlContext. The returned sender includes previous,
         * even when this rank has no active exchanges. It must be started before another communication is created.
         */
        auto communication(caravan::Event previous = {})
        {
            return caravan::defer(
                [this, previous = std::move(previous)]() mutable
                {
                    auto& device = Environment<>::get().DeviceContext();
                    auto makeReceive = [this, &device, previous](uint32_t exchange)
                    {
                        auto predecessor = receiveCompletions[exchange];
                        caravan::EventSource completion;
                        receiveCompletions[exchange] = completion.event();
                        auto branch = caravan::alpaka::withDevice(
                            device,
                            caravan::whenAll(caravan::asSender(previous), caravan::asSender(std::move(predecessor)))
                                | caravan::sequence(receive(exchange)));
                        return caravan::trackCompletion(std::move(branch), std::move(completion));
                    };
                    auto makeSend = [this, &device, previous](uint32_t exchange)
                    {
                        auto predecessor = sendCompletions[exchange];
                        caravan::EventSource completion;
                        sendCompletions[exchange] = completion.event();
                        auto branch = caravan::alpaka::withDevice(
                            device,
                            caravan::whenAll(caravan::asSender(previous), caravan::asSender(std::move(predecessor)))
                                | caravan::sequence(send(exchange)));
                        return caravan::trackCompletion(std::move(branch), std::move(completion));
                    };

                    using ReceiveBranch = decltype(makeReceive(0u));
                    using SendBranch = decltype(makeSend(0u));
                    std::vector<ReceiveBranch> receives;
                    std::vector<SendBranch> sends;
                    receives.reserve(maxExchange);
                    sends.reserve(maxExchange);
                    for(uint32_t i = 0; i < maxExchange; ++i)
                    {
                        if(hasReceiveExchange(i))
                            receives.push_back(makeReceive(i));

                        auto const sendEx = Mask::getMirroredExchangeType(i);
                        if(hasSendExchange(sendEx))
                            sends.push_back(makeSend(sendEx));
                    }
                    return caravan::whenAll(
                        caravan::asSender(std::move(previous)),
                        caravan::whenAll(std::move(receives)),
                        caravan::whenAll(std::move(sends)));
                });
        }

        /** Submit CORE and receive copies independently, then enqueue BORDER behind their native fences.
         *
         * Send retirement remains part of terminal completion. Receive and CORE completion events are also joined
         * for error reporting and lifetime retirement, but they no longer gate BORDER submission on host fence
         * observation. The border factory is invoked once all receive-copy fences have been recorded.
         */
        template<typename T_Core, typename T_BorderFactory>
        auto communicationThen(T_Core core, T_BorderFactory borderFactory, caravan::Event previous = {})
        {
            return caravan::defer(
                [this,
                 core = std::move(core),
                 borderFactory = std::move(borderFactory),
                 previous = std::move(previous)]() mutable
                {
                    using Queue = ComputeDeviceQueue;
                    using Work = caravan::alpaka::SubmittedWork<Queue>;
                    auto& device = Environment<>::get().DeviceContext();

                    std::size_t receiveCount = 0u;
                    for(uint32_t i = 0; i < maxExchange; ++i)
                        receiveCount += hasReceiveExchange(i) ? 1u : 0u;
                    auto receiveWork = std::make_shared<std::vector<std::optional<Work>>>(receiveCount);

                    auto makeReceive = [this, previous, receiveWork](uint32_t exchange, std::size_t index)
                    {
                        auto predecessor = receiveCompletions[exchange];
                        return caravan::whenAll(
                                   caravan::asSender(previous),
                                   caravan::asSender(std::move(predecessor)))
                               | caravan::sequence(receiveSubmitted(exchange))
                               | caravan::then(
                                   [this, receiveWork, exchange, index](auto submitted) mutable
                                   {
                                       receiveCompletions[exchange] = submitted.deviceWork.completion();
                                       (*receiveWork)[index].emplace(std::move(submitted.deviceWork));
                                   });
                    };
                    auto makeSend = [this, &device, previous](uint32_t exchange)
                    {
                        auto predecessor = sendCompletions[exchange];
                        caravan::EventSource completion;
                        sendCompletions[exchange] = completion.event();
                        auto branch = caravan::alpaka::withDevice(
                            device,
                            caravan::whenAll(
                                caravan::asSender(previous),
                                caravan::asSender(std::move(predecessor)))
                                | caravan::sequence(send(exchange)));
                        return caravan::trackCompletion(std::move(branch), std::move(completion));
                    };

                    using ReceiveBranch = decltype(makeReceive(0u, 0u));
                    using SendBranch = decltype(makeSend(0u));
                    std::vector<ReceiveBranch> receives;
                    std::vector<SendBranch> sends;
                    receives.reserve(receiveCount);
                    sends.reserve(maxExchange);
                    std::size_t receiveIndex = 0u;
                    for(uint32_t i = 0; i < maxExchange; ++i)
                    {
                        if(hasReceiveExchange(i))
                            receives.push_back(makeReceive(i, receiveIndex++));

                        auto const sendEx = Mask::getMirroredExchangeType(i);
                        if(hasSendExchange(sendEx))
                            sends.push_back(makeSend(sendEx));
                    }

                    auto ready = caravan::whenAll(
                        caravan::alpaka::startSubmission(device, std::move(core)),
                        caravan::whenAll(std::move(receives)));
                    auto border = std::move(ready)
                                  | caravan::letValue(
                                      [receiveWork,
                                       &device,
                                       borderFactory = std::move(borderFactory)](Work& coreWork) mutable
                                      {
                                          std::vector<caravan::EventSender> retirements;
                                          retirements.reserve(receiveWork->size() + 1u);
                                          retirements.push_back(caravan::asSender(coreWork.completion()));
                                          for(auto const& work : *receiveWork)
                                          {
                                              if(!work)
                                                  throw std::logic_error("Receive dependency was not published");
                                              retirements.push_back(caravan::asSender(work->completion()));
                                          }

                                          auto waits = caravan::alpaka::submit(
                                              [coreWork, receiveWork](auto& queue)
                                              {
                                                  coreWork.waitOn(queue);
                                                  for(auto const& work : *receiveWork)
                                                      work->waitOn(queue);
                                              });
                                          auto deviceBorder = caravan::alpaka::withDevice(
                                              device,
                                              std::move(waits)
                                                  | caravan::alpaka::sequence(std::invoke(borderFactory)));
                                          return caravan::whenAll(
                                              std::move(deviceBorder),
                                              caravan::whenAll(std::move(retirements)));
                                      });
                    // Start CORE/receives first; eager send construction must not delay the CORE launch.
                    return caravan::whenAll(
                        std::move(border),
                        caravan::whenAll(std::move(sends)),
                        caravan::asSender(std::move(previous)));
                });
        }

        /**
         * Returns the GridLayout describing this GridBuffer.
         *
         * @return the layout of this buffer
         */
        GridLayout<DIM> getGridLayout()
        {
            return gridLayout;
        }

    protected:
        /*if we have one exchange we don't check if communicationTag has been used before*/
        bool hasOneExchange;
        uint32_t lastUsedCommunicationTag;
        GridLayout<DIM> gridLayout;

        Mask sendMask;
        Mask receiveMask;

        std::unique_ptr<Exchange<BORDERTYPE, DIM>> sendExchanges[27];
        std::unique_ptr<Exchange<BORDERTYPE, DIM>> receiveExchanges[27];
        caravan::Event receiveCompletions[27];
        caravan::Event sendCompletions[27];

        uint32_t maxExchange; // use max exchanges and run over the array is faster as use set from stl
    };

} // namespace pmacc
