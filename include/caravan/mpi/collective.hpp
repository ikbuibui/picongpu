/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <exception>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include <caravan/core/sender.hpp>
#include <caravan/mpi/context.hpp>

namespace caravan::mpi
{
    namespace collective_detail
    {
        template<typename T_Sender, typename T_Factory, typename T_Receiver>
        class ManagedCollectiveOperation
        {
            struct PredecessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->prepareSuccessor(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->release([owner = owner, error = std::move(error)]() mutable noexcept
                                   { owner->m_receiver.set_error(std::move(error)); });
                }

                void set_stopped() noexcept
                {
                    owner->release([owner = owner]() noexcept { owner->m_receiver.set_stopped(); });
                }

                ManagedCollectiveOperation* owner;
            };

            struct SuccessorReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->m_receiver.set_value(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->m_receiver.set_error(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->m_receiver.set_stopped();
                }

                ManagedCollectiveOperation* owner;
            };

            using SuccessorSender = caravan::detail::SuccessorSender<T_Sender, T_Factory>;

            class SuccessorOperation
            {
            public:
                SuccessorOperation(SuccessorSender sender, ManagedCollectiveOperation* owner)
                    : m_operation(std::move(sender).connect(SuccessorReceiver{owner}))
                {
                }

                void start() noexcept
                {
                    m_operation.start();
                }

            private:
                decltype(std::declval<SuccessorSender&&>().connect(std::declval<SuccessorReceiver>())) m_operation;
            };

        public:
            ManagedCollectiveOperation(
                MpiContext& context,
                caravan::detail::ManagedCollectiveTicket ticket,
                T_Sender sender,
                T_Factory factory,
                T_Receiver receiver)
                : m_context(&context)
                , m_ticket(ticket)
                , m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
                , m_predecessor(std::move(sender).connect(PredecessorReceiver{this}))
            {
            }

            ManagedCollectiveOperation(ManagedCollectiveOperation const&) = delete;
            ManagedCollectiveOperation& operator=(ManagedCollectiveOperation const&) = delete;
            ManagedCollectiveOperation(ManagedCollectiveOperation&&) = delete;
            ManagedCollectiveOperation& operator=(ManagedCollectiveOperation&&) = delete;

            void start() & noexcept
            {
                m_predecessor.start();
            }

        private:
            template<typename... T>
            void prepareSuccessor(T&&... values) noexcept
            {
                try
                {
                    m_successor.emplace(std::invoke(m_factory, std::forward<T>(values)...), this);
                    release([this]() noexcept { m_successor->start(); });
                }
                catch(...)
                {
                    auto error = std::current_exception();
                    release([this, error = std::move(error)]() mutable noexcept
                            { m_receiver.set_error(std::move(error)); });
                }
            }

            void release(std::function<void()> start) noexcept
            {
                try
                {
                    caravan::detail::CollectiveAccess::release(*m_context, m_ticket, std::move(start));
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            MpiContext* m_context;
            caravan::detail::ManagedCollectiveTicket m_ticket;
            T_Factory m_factory;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<PredecessorReceiver>())) m_predecessor;
            std::optional<SuccessorOperation> m_successor;
        };

        template<typename T_Sender, typename T_Factory>
        class ManagedCollectiveSender
        {
        public:
            using completion_signatures
                = CompletionSignaturesOf<caravan::detail::SuccessorSender<T_Sender, T_Factory>>;

            ManagedCollectiveSender(
                MpiContext& context,
                caravan::detail::ManagedCollectiveTicket ticket,
                T_Sender sender,
                T_Factory factory)
                : m_context(&context)
                , m_ticket(ticket)
                , m_sender(std::move(sender))
                , m_factory(std::move(factory))
            {
            }

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return ManagedCollectiveOperation<T_Sender, T_Factory, std::decay_t<T_Receiver>>{
                    *m_context,
                    m_ticket,
                    std::move(m_sender),
                    std::move(m_factory),
                    std::forward<T_Receiver>(receiver)};
            }

        private:
            MpiContext* m_context;
            caravan::detail::ManagedCollectiveTicket m_ticket;
            T_Sender m_sender;
            T_Factory m_factory;
        };
    } // namespace collective_detail

    /** Plan collective initiation order independently of predecessor readiness.
     *
     * Every rank must submit and start the same sequence on this communicator.
     * Each factory must return exactly one collective sender for that communicator.
     * Failed/stopped predecessors retire their entry without initiating MPI.
     */
    class CollectiveLane
    {
    public:
        CollectiveLane(MpiContext& context, CommunicatorId communicator = worldCommunicator)
            : m_context(&context)
            , m_communicator(communicator)
        {
        }

        template<Sender T_Sender, typename T_Factory>
        auto submit(T_Sender sender, T_Factory factory)
        {
            auto const ticket = caravan::detail::CollectiveAccess::reserve(*m_context, m_communicator);
            return collective_detail::ManagedCollectiveSender<T_Sender, std::decay_t<T_Factory>>{
                *m_context,
                ticket,
                std::move(sender),
                std::move(factory)};
        }

    private:
        MpiContext* m_context;
        CommunicatorId m_communicator;
    };
} // namespace caravan::mpi
