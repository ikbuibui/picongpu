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

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<typename T_Factory, typename T_Receiver>
        class DeferOperation
        {
            using Sender = std::invoke_result_t<T_Factory&>;
            using Successor = ConnectedOperation<Sender, T_Receiver>;

        public:
            DeferOperation(T_Factory factory, T_Receiver receiver)
                : m_factory(std::move(factory))
                , m_receiver(std::move(receiver))
            {
            }

            DeferOperation(DeferOperation const&) = delete;
            DeferOperation& operator=(DeferOperation const&) = delete;
            DeferOperation(DeferOperation&&) = delete;
            DeferOperation& operator=(DeferOperation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    m_successor.emplace(std::invoke(m_factory), m_receiver);
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                    return;
                }
                // Completion may destroy this operation. Do not access members after start().
                m_successor->start();
            }

        private:
            T_Factory m_factory;
            T_Receiver m_receiver;
            std::optional<Successor> m_successor;
        };
    } // namespace detail

    /** Lazily invoke a nullary sender factory when the connected operation starts. */
    template<typename T_Factory>
    class DeferSender
    {
        using Sender = std::invoke_result_t<T_Factory&>;
        static_assert(caravan::Sender<Sender>);

    public:
        using completion_signatures = CompletionSignaturesOf<Sender>;

        explicit DeferSender(T_Factory factory) : m_factory(std::move(factory))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::DeferOperation<T_Factory, std::decay_t<T_Receiver>>{
                std::move(m_factory),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Factory m_factory;
    };

    template<typename T_Factory>
    auto defer(T_Factory factory)
    {
        return DeferSender<T_Factory>{std::move(factory)};
    }
} // namespace caravan
