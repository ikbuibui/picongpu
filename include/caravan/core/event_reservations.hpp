/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <exception>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/core/eager.hpp>

namespace caravan
{
    /** Transactional replacement of resource-retirement events during lazy graph setup.
     *
     * Setup on a given resource must be serialized. Setters must not throw, and their targets must outlive setup.
     * Replacements are rolled back unless the entire graph connects successfully, before any child starts.
     */
    class EventReservations
    {
    public:
        EventReservations() = default;
        EventReservations(EventReservations const&) = delete;
        EventReservations& operator=(EventReservations const&) = delete;

        ~EventReservations()
        {
            rollback();
        }

        template<typename T_Setter>
        Event replace(Event previous, Event next, T_Setter setter)
        {
            static_assert(std::is_nothrow_invocable_v<T_Setter&, Event>);
            // Allocate the undo record before publishing the replacement.
            m_undo.emplace_back([setter, previous]() mutable { setter(std::move(previous)); });
            setter(std::move(next));
            return previous;
        }

        Event replace(Event& target, Event next)
        {
            return replace(target, std::move(next), [&target](Event event) noexcept { target = std::move(event); });
        }

        void commit() noexcept
        {
            m_undo.clear();
        }

        void rollback() noexcept
        {
            while(!m_undo.empty())
            {
                m_undo.back()();
                m_undo.pop_back();
            }
        }

    private:
        std::vector<std::function<void()>> m_undo;
    };

    namespace detail
    {
        template<typename T_Factory, typename T_Receiver>
        class ReservedDeferOperation
        {
            using Sender = std::invoke_result_t<T_Factory&, EventReservations&>;
            using Successor = ConnectedOperation<Sender, T_Receiver>;

        public:
            ReservedDeferOperation(T_Factory factory, T_Receiver receiver)
                : m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
            {
            }

            ReservedDeferOperation(ReservedDeferOperation const&) = delete;
            ReservedDeferOperation& operator=(ReservedDeferOperation const&) = delete;
            ReservedDeferOperation(ReservedDeferOperation&&) = delete;
            ReservedDeferOperation& operator=(ReservedDeferOperation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    m_successor.emplace(std::invoke(m_factory, m_reservations), m_receiver);
                }
                catch(...)
                {
                    // Restore resource state before error delivery can start a retry.
                    m_reservations.rollback();
                    m_receiver.set_error(std::current_exception());
                    return;
                }
                m_reservations.commit();
                m_successor->start();
            }

        private:
            T_Factory m_factory;
            T_Receiver m_receiver;
            EventReservations m_reservations;
            std::optional<Successor> m_successor;
        };
    } // namespace detail

    template<typename T_Factory>
    class ReservedDeferSender
    {
        using Sender = std::invoke_result_t<T_Factory&, EventReservations&>;

    public:
        using completion_signatures = CompletionSignaturesOf<Sender>;

        explicit ReservedDeferSender(T_Factory factory) : m_factory(std::move(factory))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ReservedDeferOperation<T_Factory, std::decay_t<T_Receiver>>{
                std::move(m_factory),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Factory m_factory;
    };

    /** Defer graph construction with retirement reservations committed only after successful connection. */
    template<typename T_Factory>
    auto deferWithReservations(T_Factory factory)
    {
        return ReservedDeferSender<T_Factory>{std::move(factory)};
    }
} // namespace caravan
