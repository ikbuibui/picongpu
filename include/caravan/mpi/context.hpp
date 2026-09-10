/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <caravan/core/retained.hpp>

namespace caravan
{
    namespace detail
    {
        struct NativeAccess;
        struct NativeInvocation;
        struct NativeSubmission;
    } // namespace detail

    struct CommunicatorId
    {
        std::uint32_t value;

        friend bool operator==(CommunicatorId const&, CommunicatorId const&) = default;
    };

    inline constexpr CommunicatorId worldCommunicator{0u};

    class MpiContext;
    class MpiExternalRuntime;
    class MpiRuntime;

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

    namespace detail
    {
        template<typename T_Byte>
        struct MpiBufferView : Retained<std::span<T_Byte>, std::shared_ptr<void>>
        {
            using Base = Retained<std::span<T_Byte>, std::shared_ptr<void>>;
            using Base::Base;

            MpiBufferView(Base buffer) : Base(std::move(buffer))
            {
            }

            template<typename T, std::size_t T_Extent>
            requires std::convertible_to<std::span<T, T_Extent>, std::span<T_Byte>>
            MpiBufferView(std::span<T, T_Extent> bytes) : Base(bytes, {})
            {
            }
        };
    } // namespace detail

    /** MPI byte spans with optional ownership retained until native work completes.
     *
     * Plain spans borrow storage: the caller must keep it valid from sender construction
     * through completion, including failure cleanup. Do not modify send storage or access
     * receive storage while MPI may use it. Use retain(span, owner) to keep an owner alive;
     * retention does not prevent conflicting accesses.
     */
    using ConstMpiBuffer = detail::MpiBufferView<std::byte const>;
    using MpiBuffer = detail::MpiBufferView<std::byte>;

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
        MpiContext();

        void run();
        bool progress();
        void requestShutdown();
        bool shutdownComplete() const noexcept;
        void submitNative(detail::NativeSubmission submission);
        void invokeNative(detail::NativeInvocation submission);

        std::unique_ptr<Impl> m_implementation;

        detail::ManagedCollectiveTicket reserveManagedCollective(CommunicatorId communicator);
        void releaseManagedCollective(detail::ManagedCollectiveTicket ticket, std::function<void()> start);
        void abandonManagedCollective(detail::ManagedCollectiveTicket ticket) noexcept;

        friend class MpiRuntime;
        friend class MpiExternalRuntime;
        friend struct detail::CollectiveAccess;
        friend struct detail::NativeAccess;
    };

} // namespace caravan
