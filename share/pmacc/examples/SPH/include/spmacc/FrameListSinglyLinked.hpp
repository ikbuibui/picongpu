/* Copyright 2013-2024 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#include "spmacc/Frame.hpp"

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif
#include <pmacc/lockstep.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/frame_types.hpp>
#include <pmacc/particles/memory/dataTypes/FramePointer.hpp>
#include <pmacc/traits/IsSpecializationOf.hpp>
#include <pmacc/verify.hpp>

namespace pmacc
{
    namespace sph
    {
        /**
         * A singly-linked list holding frames with particle data.
         *
         * @tparam T_Frame datatype for frames`
         * @tparam T_DeviceHeapHandle device heap handle type
         */
        template<concepts::SpecializationOf<Frame> T_Frame, typename T_DeviceHeapHandle>
        class SingleLinkedFrameList
        {
        private:
            PMACC_ALIGN(m_deviceHeapHandle, T_DeviceHeapHandle);
            PMACC_ALIGN(m_firstFrame, FramePointer<T_Frame>);
            PMACC_ALIGN(m_lastFrame, FramePointer<T_Frame>);
            PMACC_ALIGN(hostMemoryOffset, int64_t) { 0 };

        public:
            using FrameType = T_Frame;
            using FramePtr = FramePointer<FrameType>;
            using DeviceHeapHandle = T_DeviceHeapHandle;

            static constexpr uint32_t frameSize = FrameType::NumSlots::value;

            /** default constructor
             *
             * \warning after this call the object is in a invalid state and must be
             * initialized with an assignment of a valid SingleLinkedFrameList
             */
            HDINLINE SingleLinkedFrameList() = default;

            HDINLINE SingleLinkedFrameList(DeviceHeapHandle const& deviceHeapHandle)
                : m_deviceHeapHandle(deviceHeapHandle)
                , m_firstFrame()
                , m_lastFrame()
            {
            }

            HDINLINE SingleLinkedFrameList(DeviceHeapHandle const& deviceHeapHandle, int64_t memoryOffset)
                : m_deviceHeapHandle(deviceHeapHandle)
                , m_firstFrame()
                , m_lastFrame()
                , hostMemoryOffset(memoryOffset)
            {
            }

            /**
             * Returns an empty frame from data heap.
             *
             * @return an empty frame
             */
            template<typename T_Worker>
            DINLINE FramePtr getEmptyFrame(T_Worker const& worker)
            {
                FrameType* tmp = nullptr;
                int const maxTries = 13;
                for(int numTries = 0; numTries < maxTries; ++numTries)
                {
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
                    tmp = (FrameType*) m_deviceHeapHandle.malloc(worker.getAcc(), sizeof(FrameType));
#else
                    tmp = new FrameType;
#endif
                    if(tmp != nullptr)
                    {
                        /* disable all particles since we can not assume that newly allocated memory contains zeros */
                        for(int i = 0; i < static_cast<int>(FrameType::frameSize); ++i)
                            (*tmp)[i][multiMask_] = 0;
                        /* takes care that changed values are visible to all threads inside this block*/
                        alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Block{});
                        break;
                    }
                }

                PMACC_DEVICE_VERIFY_MSG(
                    tmp != nullptr,
                    "Error: Out of device heap memory in %s:%u\n",
                    __FILE__,
                    __LINE__);

                return FramePtr(tmp);
            }

            /**
             * Removes frame from heap data heap.
             *
             * @param frame frame to remove
             */
            template<typename T_Worker>
            DINLINE void removeFrame(T_Worker const& worker, FramePtr& frame)
            {
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
                m_deviceHeapHandle.free(worker.getAcc(), (void*) frame.ptr);
#else
                delete(frame.ptr);
#endif
                frame.ptr = nullptr;
            }

            HDINLINE
            FramePtr mapPtr(FramePtr devPtr) const
            {
#if (PMACC_DEVICE_COMPILE == 1)
                return devPtr;
#else
                int64_t useOffset = hostMemoryOffset * static_cast<int64_t>(devPtr.ptr != nullptr);
                return FramePtr(reinterpret_cast<FrameType*>(reinterpret_cast<char*>(devPtr.ptr) - useOffset));
#endif
            }

            /**
             * Returns the next frame in the linked list.
             *
             * @param frame the active frame
             * @return the next frame in the list
             */
            HDINLINE FramePtr getNextFrame(FramePtr const& frame) const
            {
                return mapPtr(frame->nextFrame);
            }

            /**
             * Returns the last frame of the list.
             *
             * @return the last frame of the linked list
             */
            HDINLINE FramePtr getLastFrame() const
            {
                return mapPtr(m_lastFrame);
            }

            /**
             * Returns the first frame of the list.
             *
             * @return the first frame of the linked list
             */
            HDINLINE FramePtr getFirstFrame() const
            {
                return mapPtr(m_firstFrame);
            }

            /**
             * Sets frame as the first frame of the list.
             *
             * @param frame frame to set as first frame
             */
            template<typename T_Worker>
            DINLINE void setAsFirstFrame(T_Worker const& worker, FramePtr& frame)
            {
                frame->nextFrame = m_firstFrame;

                /* takes care that nextFrame is visible to all threads on the gpu */
                alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});

                FramePtr oldFirstFramePtr((FrameType*) alpaka::atomicExch(
                    worker.getAcc(),
                    (unsigned long long int*) &m_firstFrame.ptr,
                    (unsigned long long int) frame.ptr,
                    ::alpaka::hierarchy::Grids{}));

                frame->nextFrame = oldFirstFramePtr;
                if(!oldFirstFramePtr.isValid())
                {
                    // we add the first frame to an empty list
                    m_lastFrame = frame;
                }
            }

            /**
             * Sets frame as the last frame of the list.
             *
             * @param frame frame to set as last frame
             */
            template<typename T_Worker>
            DINLINE void setAsLastFrame(T_Worker const& worker, FramePtr& frame)
            {
                frame->nextFrame = FramePtr();

                /* takes care that nextFrame is visible to all threads on the gpu */
                alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});

                FramePtr oldLastFramePtr((FrameType*) alpaka::atomicExch(
                    worker.getAcc(),
                    (unsigned long long int*) &m_lastFrame.ptr,
                    (unsigned long long int) frame.ptr,
                    ::alpaka::hierarchy::Grids{}));

                if(oldLastFramePtr.isValid())
                {
                    oldLastFramePtr->nextFrame = frame;
                }
                else
                {
                    // we add the first frame to an empty list
                    m_firstFrame = frame;
                }
            }

            /**
             * Removes the first frame of the list.
             * This call is not threadsafe, only one thread may call this function.
             * @return true if more frames in list, else false
             */
            template<typename T_Worker>
            DINLINE bool removeFirstFrame(T_Worker const& worker)
            {
                FramePtr first = m_firstFrame;
                if(first.isValid())
                {
                    FramePtr next = first->nextFrame;
                    m_firstFrame = next;

                    if(!next.isValid())
                    {
                        // we removed the last frame
                        m_lastFrame = FramePtr();
                    }

                    removeFrame(worker, first);
                    return next.isValid();
                }
                return false;
            }

            /**
             * Removes the last frame of the list.
             * This call is not threadsafe and requires traversal of the entire list.
             * @return true if more frames in list, else false
             */
            template<typename T_Worker>
            DINLINE bool removeLastFrame(T_Worker const& worker)
            {
                FramePtr last = m_lastFrame;
                if(!last.isValid())
                    return false;

                // For singly-linked list, we need to traverse to find the previous frame
                if(m_firstFrame == last)
                {
                    // Only one frame in the list
                    m_firstFrame = FramePtr();
                    m_lastFrame = FramePtr();
                    removeFrame(worker, last);
                    return false;
                }

                // Find the second-to-last frame
                FramePtr current = m_firstFrame;
                while(current.isValid() && current->nextFrame != last)
                {
                    current = current->nextFrame;
                }

                if(current.isValid())
                {
                    current->nextFrame = FramePtr();
                    m_lastFrame = current;
                    removeFrame(worker, last);
                    return true;
                }

                return false;
            }

            /**
             * Lock-free push frame to the front of the list.
             * Thread-safe and can be called concurrently.
             *
             * @param frame frame to push to front
             */
            template<typename T_Worker>
            DINLINE void pushFront(T_Worker const& worker, FramePtr& frame)
            {
                FramePtr expected;
                do
                {
                    expected = m_firstFrame;
                    frame->nextFrame = expected;
                    alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});
                } while(!alpaka::atomicCas(
                    worker.getAcc(),
                    (unsigned long long int*) &m_firstFrame.ptr,
                    (unsigned long long int) expected.ptr,
                    (unsigned long long int) frame.ptr,
                    ::alpaka::hierarchy::Grids{}));

                // Update last frame pointer if list was empty
                if(!expected.isValid())
                {
                    m_lastFrame = frame;
                }
            }

            /**
             * Lock-free pop frame from the front of the list.
             * Thread-safe but only one thread should successfully pop.
             *
             * @return popped frame or invalid FramePtr if list is empty
             */
            template<typename T_Worker>
            DINLINE FramePtr popFront(T_Worker const& worker)
            {
                FramePtr expected;
                FramePtr next;

                do
                {
                    expected = m_firstFrame;
                    if(!expected.isValid())
                        return FramePtr(); // List is empty

                    next = expected->nextFrame;
                } while(!alpaka::atomicCas(
                    worker.getAcc(),
                    (unsigned long long int*) &m_firstFrame.ptr,
                    (unsigned long long int) expected.ptr,
                    (unsigned long long int) next.ptr,
                    ::alpaka::hierarchy::Grids{}));

                // Update last frame pointer if list is now empty
                if(!next.isValid())
                {
                    m_lastFrame = FramePtr();
                }

                // Clear the popped frame's next pointer
                expected->nextFrame = FramePtr();
                return expected;
            }

            /**
             * Check if the list is empty.
             * Lock-free read operation.
             *
             * @return true if empty, false otherwise
             */
            HDINLINE bool isEmpty() const
            {
                return !m_firstFrame.isValid();
            }

            /**
             * Traverse the list and apply a function to each frame.
             * Lock-free read operation, but list should not be modified during traversal.
             *
             * @tparam T_Functor functor type with signature void(FramePtr&)
             * @param functor function to apply to each frame
             */
            template<typename T_Functor>
            HDINLINE void traverse(T_Functor&& functor) const
            {
                FramePtr current = m_firstFrame;
                while(current.isValid())
                {
                    functor(current);
                    current = current->nextFrame;
                }
            }

            /**
             * Traverse the list with index and apply a function to each frame.
             * Lock-free read operation, but list should not be modified during traversal.
             *
             * @tparam T_Functor functor type with signature void(FramePtr&, uint32_t idx)
             * @param functor function to apply to each frame with its index
             */
            template<typename T_Functor>
            HDINLINE void traverseWithIndex(T_Functor&& functor) const
            {
                FramePtr current = m_firstFrame;
                uint32_t idx = 0;
                while(current.isValid())
                {
                    functor(current, idx);
                    current = current->nextFrame;
                    ++idx;
                }
            }

            /**
             * Count the number of frames in the list.
             * Not atomic - use only when no concurrent modifications occur.
             *
             * @return number of frames in the list
             */
            HDINLINE uint32_t count() const
            {
                uint32_t cnt = 0;
                FramePtr current = m_firstFrame;
                while(current.isValid())
                {
                    ++cnt;
                    current = current->nextFrame;
                }
                return cnt;
            }

            /**
             * Find a frame matching a predicate.
             * Lock-free read operation.
             *
             * @tparam T_Predicate predicate type with signature bool(FramePtr const&)
             * @param predicate function returning true for matching frame
             * @return first matching frame or invalid FramePtr if not found
             */
            template<typename T_Predicate>
            HDINLINE FramePtr find(T_Predicate&& predicate) const
            {
                FramePtr current = m_firstFrame;
                while(current.isValid())
                {
                    if(predicate(current))
                        return current;
                    current = current->nextFrame;
                }
                return FramePtr();
            }

            /**
             * Lock-free remove of a specific frame (if it's the first frame).
             * For general removal, use removeAfter with the previous frame.
             * Thread-safe.
             *
             * @param frame frame to remove (must be first frame)
             * @return true if successfully removed, false otherwise
             */
            template<typename T_Worker>
            DINLINE bool removeIfFirst(T_Worker const& worker, FramePtr const& frame)
            {
                if(!frame.isValid())
                    return false;

                bool success = alpaka::atomicCas(
                    worker.getAcc(),
                    (unsigned long long int*) &m_firstFrame.ptr,
                    (unsigned long long int) frame.ptr,
                    (unsigned long long int) frame->nextFrame.ptr,
                    ::alpaka::hierarchy::Grids{});

                if(success)
                {
                    // Update last frame pointer if list is now empty
                    if(!frame->nextFrame.isValid())
                    {
                        m_lastFrame = FramePtr();
                    }
                    frame->nextFrame = FramePtr();
                }

                return success;
            }

            /**
             * Remove the frame after the given frame.
             * Not lock-free - caller must ensure no concurrent modifications to this part of the list.
             *
             * @param prevFrame frame before the one to remove
             * @return removed frame or invalid FramePtr if prevFrame was last
             */
            template<typename T_Worker>
            DINLINE FramePtr removeAfter(T_Worker const& worker, FramePtr& prevFrame)
            {
                if(!prevFrame.isValid())
                    return FramePtr();

                FramePtr toRemove = prevFrame->nextFrame;
                if(!toRemove.isValid())
                    return FramePtr();

                prevFrame->nextFrame = toRemove->nextFrame;

                // Update last frame pointer if we removed the last frame
                if(!toRemove->nextFrame.isValid())
                {
                    m_lastFrame = prevFrame;
                }

                toRemove->nextFrame = FramePtr();
                return toRemove;
            }

            /**
             * Clear the entire list, removing and deallocating all frames.
             * Not thread-safe - should only be called when no other operations are occurring.
             */
            template<typename T_Worker>
            DINLINE void clear(T_Worker const& worker)
            {
                FramePtr current = m_firstFrame;
                while(current.isValid())
                {
                    FramePtr next = current->nextFrame;
                    removeFrame(worker, current);
                    current = next;
                }
                m_firstFrame = FramePtr();
                m_lastFrame = FramePtr();
            }
        };
    } // namespace sph

    namespace lockstep::traits
    {
        //! Specialization to create a lockstep block configuration out of a singly-linked frame list.
        template<class T_Frame, typename T_DeviceHeapHandle>
        struct MakeBlockCfg<sph::SingleLinkedFrameList<T_Frame, T_DeviceHeapHandle>> : std::true_type
        {
            static constexpr uint32_t frameSize
                = sph::SingleLinkedFrameList<T_Frame, T_DeviceHeapHandle>::FrameType::frameSize;
            using type = BlockCfg<math::CT::UInt32<frameSize>>;
        };
    } // namespace lockstep::traits

} // namespace pmacc
