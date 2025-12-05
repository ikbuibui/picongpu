#pragma once

#include <pmacc/memory/Align.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/memory/dataTypes/FramePointer.hpp>

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/mem/fence/Traits.hpp>

#include <concepts>
#include <cstdint>
#include <new>

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif

namespace pmacc
{
    namespace sph
    {
        template<typename T>
        struct Node
        {
            using NodePtr = Node<T>*;
            PMACC_ALIGN(data, T);
            PMACC_ALIGN(next, NodePtr);
        };

        /**
         * A singly-linked list of frames on the acc
         * uses new and delete on CPU and mallocMC on GPU
         *
         * @tparam T_DeviceHeapHandle device heap handle type
         */
        template<typename T, typename T_DeviceHeapHandle>
        struct SingleLinkedListDevice
        {
            using NodeType = Node<T>;
            using NodePtr = NodeType*;

            constexpr SingleLinkedListDevice(T_DeviceHeapHandle const& deviceHeapHandle)
                : m_deviceHeapHandle(deviceHeapHandle)
            {
            }

            /**
             * Returns a pointer to a free node from data heap.
             * If T is default initializable, the type is constructed after allocation, else it is not constructed
             *
             * @param worker
             */
            [[nodiscard]] constexpr NodePtr getEmptyNode(auto const& worker)
            {
                NodePtr tmp = allocateRawNode(worker);

                PMACC_DEVICE_VERIFY_MSG(
                    tmp != nullptr,
                    "Error: Out of device heap memory in %s:%u\n",
                    __FILE__,
                    __LINE__);

                if constexpr(std::default_initializable<T>)
                {
                    if(tmp)
                    {
                        new(tmp) NodeType{};
                    }
                }

                return tmp;
            }

            /**
             * Removes frame from heap data heap.
             * Takes ownership and sets the user provided ptr to nullptr
             *
             * @param worker
             * @param node pointer to node to remove
             */
            constexpr void removeNode(auto const& worker, NodePtr& node)
            {
                if(!node)
                    return;

                if constexpr(!std::is_trivially_destructible_v<T>)
                {
                    node->data.~T();
                }

#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
                m_deviceHeapHandle.free(worker.getAcc(), (void*) node);
#else
                operator delete(node, std::nothrow);
#endif
                node = nullptr;
            }

            /**
             * Thread-safe insertion of a node at the back of the list.
             * Takes ownership of the provided node.
             *
             * @param worker
             * @param node pointer to node to insert
             */
            constexpr void push_back(auto const& worker, NodePtr& node)
            {
                if(!node)
                    return;

                node->next = nullptr;

                NodePtr oldLast = alpaka::atomicExch(worker.getAcc(), &m_lastNode, node);

                if(oldLast != nullptr)
                {
                    // List was non-empty, link old last to new node
                    oldLast->next = node;
                }
                else
                {
                    // List was empty, update first node
                    m_firstNode = node;
                }
                // fence to publish changes to the list to everyone
                // TODO use a release fence
                alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});

                node = nullptr;
            }


        private:
            // Helper to abstract raw allocation and retry logic
            [[nodiscard]] constexpr NodePtr allocateRawNode(auto const& worker)
            {
                for(int i = 0; i < allocationMaxRetries; ++i)
                {
                    void* rawPtr = nullptr;
#if (BOOST_LANG_CUDA || BOOST_COMP_HIP)
                    // Explicit cast required for C++
                    rawPtr = m_deviceHeapHandle.malloc(worker.getAcc(), sizeof(NodeType));
#else
                    // Use nothrow to ensure nullptr is returned on failure,
                    // preventing exceptions from breaking the retry loop.
                    rawPtr = operator new(sizeof(NodeType), std::nothrow);
#endif
                    if(rawPtr != nullptr)
                    {
                        return static_cast<NodePtr>(rawPtr);
                    }
                }
                return nullptr;
            }

        private:
            static constexpr int allocationMaxRetries = 13;
            PMACC_ALIGN(m_deviceHeapHandle, T_DeviceHeapHandle);
            PMACC_ALIGN(m_firstNode, NodePtr) { nullptr };
            PMACC_ALIGN(m_lastNode, NodePtr) { nullptr };
        };
    } // namespace sph
} // namespace pmacc
