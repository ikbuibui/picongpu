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

    class MpiContext;

    namespace detail
    {
        struct ManagedCollectiveTicket
        {
            CommunicatorId communicator;
            std::size_t sequence;
        };

        struct CollectiveAccess
        {
            static ManagedCollectiveTicket reserve(MpiContext& context, CommunicatorId communicator);
            static void release(MpiContext& context, ManagedCollectiveTicket ticket, std::function<void()> start);
            static void abandon(MpiContext& context, ManagedCollectiveTicket ticket) noexcept;
        };
    } // namespace detail

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

    /** MPI buffer view with optional explicitly retained ownership.
     *
     * A borrowed buffer must remain valid until the operation completes. Passing
     * an owner retains that allocation in the operation and native request state.
     */
    class BufferLease
    {
    public:
        BufferLease(std::shared_ptr<void> allocation, void* data, std::size_t bytes)
            : m_allocation(std::move(allocation))
            , m_data(data)
            , m_bytes(bytes)
        {
        }

        static BufferLease borrowed(void* data, std::size_t bytes)
        {
            return BufferLease{{}, data, bytes};
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
            return m_bytes == 0u || m_data != nullptr;
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

    struct ReduceResult
    {
        std::size_t elements;
    };

    struct GatherResult
    {
        std::size_t bytes;
    };

    struct CommunicatorInfo
    {
        CommunicatorId communicator;
        int rank;
        int size;
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

    /** MPI backend authority owning lifecycle, progress, and native resources.
     *
     * This context is not a scheduler for application continuations.
     */
    class MpiContext
    {
    public:
        MpiContext(MpiContext const&) = delete;
        MpiContext& operator=(MpiContext const&) = delete;
        ~MpiContext();

        TopologySnapshot topology() const;

    private:
        class Impl;
        explicit MpiContext(std::unique_ptr<Impl> implementation);

        void run();
        void requestShutdown();
        void submitNative(detail::NativeSubmission submission);
        void invokeBlocking(detail::NativeBlockingSubmission submission);

        std::unique_ptr<Impl> m_implementation;

        detail::ManagedCollectiveTicket reserveManagedCollective(CommunicatorId communicator);
        void releaseManagedCollective(detail::ManagedCollectiveTicket ticket, std::function<void()> start);
        void abandonManagedCollective(detail::ManagedCollectiveTicket ticket) noexcept;

        friend class MpiRuntime;
        friend struct detail::CollectiveAccess;
        friend struct detail::NativeAccess;
    };

    class MpiRuntime
    {
    public:
        template<typename T_Application>
        static int run(int& argc, char**& argv, T_Application&& application)
        {
            auto invoke = [&application](MpiContext& context)
            {
                if constexpr(std::is_invocable_v<T_Application&, MpiContext&>)
                {
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&, MpiContext&>>)
                    {
                        std::invoke(application, context);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application, context));
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
        static int runImpl(int& argc, char**& argv, std::function<int(MpiContext&)> application);
    };
} // namespace caravan
