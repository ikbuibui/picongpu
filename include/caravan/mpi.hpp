/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/core.hpp>

namespace caravan
{
    namespace detail
    {
        struct NativeAccess;
        struct NativeBlockingSubmission;
        struct NativeSubmission;
    } // namespace detail

    struct CommunicatorId
    {
        std::uint32_t value;

        friend bool operator==(CommunicatorId const&, CommunicatorId const&) = default;
    };

    inline constexpr CommunicatorId worldCommunicator{0u};

    struct Peer
    {
        int value;
        bool any = false;
    };

    inline constexpr Peer anyPeer{0, true};

    struct MessageTag
    {
        int value;
        bool any = false;
    };

    inline constexpr MessageTag anyMessageTag{0, true};

    class BufferLease
    {
    public:
        BufferLease(std::shared_ptr<void> allocation, void* data, std::size_t bytes)
            : m_allocation(std::move(allocation))
            , m_data(data)
            , m_bytes(bytes)
        {
        }

        void* data() const noexcept
        {
            return m_data;
        }

        std::size_t bytes() const noexcept
        {
            return m_bytes;
        }

        std::shared_ptr<void> lifetime() const noexcept
        {
            return m_allocation;
        }

        bool valid() const noexcept
        {
            return m_bytes == 0u || (m_allocation && m_data != nullptr);
        }

    private:
        std::shared_ptr<void> m_allocation;
        void* m_data;
        std::size_t m_bytes;
    };

    enum class ScalarType : std::uint8_t
    {
        int32,
        uint32,
        int64,
        uint64,
        float32,
        float64
    };

    enum class ReduceOperation : std::uint8_t
    {
        sum,
        minimum,
        maximum,
        product
    };

    struct SendResult
    {
        std::size_t bytes;
    };

    struct ReceiveResult
    {
        Peer source;
        MessageTag tag;
        std::size_t bytes;
    };

    struct AllReduceResult
    {
        std::size_t elements;
    };

    struct TopologySnapshot
    {
        int rank;
        int size;
        int hostLocalRank;
        CommunicatorId communicator;
        std::vector<int> dimensions;
        std::vector<int> coordinates;
        std::vector<bool> periodic;
        // Negative then positive neighbor for each dimension; -1 means no neighbor.
        std::vector<int> neighbors;
    };

    class MpiExecutor
    {
    public:
        MpiExecutor(MpiExecutor const&) = delete;
        MpiExecutor& operator=(MpiExecutor const&) = delete;
        ~MpiExecutor();

        TopologySnapshot topology() const;

        Future<TopologySnapshot> createCartesian(
            Event predecessor,
            std::vector<int> dimensions,
            std::vector<bool> periodic);

        Event destroyCommunicator(Event predecessor, CommunicatorId communicator);

        Future<SendResult> send(
            Event dataReady,
            BufferLease buffer,
            Peer destination,
            MessageTag tag,
            CommunicatorId communicator = worldCommunicator);

        Future<ReceiveResult> receive(
            Event bufferAvailable,
            BufferLease buffer,
            Peer source,
            MessageTag tag,
            CommunicatorId communicator = worldCommunicator);

        Future<AllReduceResult> allReduce(
            Event dataReady,
            BufferLease input,
            BufferLease output,
            ScalarType type,
            ReduceOperation operation,
            CommunicatorId communicator = worldCommunicator);

        Event barrier(Event predecessor, CommunicatorId communicator = worldCommunicator);

    private:
        class Impl;
        explicit MpiExecutor(std::unique_ptr<Impl> implementation);

        void run();
        void requestShutdown();
        void submitNative(Event predecessor, detail::NativeSubmission submission);
        void invokeBlocking(Event predecessor, detail::NativeBlockingSubmission submission);

        std::unique_ptr<Impl> m_implementation;

        friend class MpiRuntime;
        friend struct detail::NativeAccess;
    };

    class MpiRuntime
    {
    public:
        template<typename T_Application>
        static int run(int& argc, char**& argv, T_Application&& application)
        {
            auto invoke = [&application](MpiExecutor& executor)
            {
                if constexpr(std::is_invocable_v<T_Application&, MpiExecutor&>)
                {
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&, MpiExecutor&>>)
                    {
                        std::invoke(application, executor);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application, executor));
                }
                else
                {
                    static_assert(std::is_invocable_v<T_Application&>);
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&>>)
                    {
                        std::invoke(application);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application));
                }
            };
            return runImpl(argc, argv, invoke);
        }

    private:
        static int runImpl(int& argc, char**& argv, std::function<int(MpiExecutor&)> application);
    };
} // namespace caravan
