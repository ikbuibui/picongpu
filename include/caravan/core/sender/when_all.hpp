/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<std::size_t T_Index, typename T_Owner>
        struct WhenAllReceiver
        {
            template<typename... T>
            void set_value(T&&... values) noexcept
            {
                owner->template setValue<T_Index>(std::forward<T>(values)...);
            }

            void set_error(std::exception_ptr error) noexcept
            {
                owner->setError(std::move(error));
            }

            void set_stopped() noexcept
            {
                owner->setStopped();
            }

            decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Owner const&>().getEnv()))
                requires requires(T_Owner const& value) { value.getEnv(); }
            {
                return owner->getEnv();
            }

            T_Owner* owner;
        };

        template<std::size_t T_Index, typename T_Owner, typename T_Sender>
        class WhenAllOperationHolder
        {
            using Receiver = WhenAllReceiver<T_Index, T_Owner>;

        public:
            WhenAllOperationHolder(T_Sender sender, T_Owner* owner)
                : m_operation(std::move(sender).connect(Receiver{owner}))
            {
            }

            void start() noexcept
            {
                m_operation.start();
            }

        private:
            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };

        template<typename T_Receiver, typename T_Indices, typename... T_Senders>
        class WhenAllOperation;

        template<typename T_Receiver, std::size_t... T_Index, typename... T_Senders>
        class WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>
            : private WhenAllOperationHolder<
                  T_Index,
                  WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>,
                  T_Senders>...
        {
            using Self = WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>;

            template<std::size_t T_I>
            using Holder = WhenAllOperationHolder<T_I, Self, std::tuple_element_t<T_I, std::tuple<T_Senders...>>>;

        public:
            WhenAllOperation(std::tuple<T_Senders...> senders, T_Receiver receiver)
                : Holder<T_Index>(std::move(std::get<T_Index>(senders)), this)...
                , m_receiver(std::move(receiver))
            {
            }

            WhenAllOperation(WhenAllOperation const&) = delete;
            WhenAllOperation& operator=(WhenAllOperation const&) = delete;
            WhenAllOperation(WhenAllOperation&&) = delete;
            WhenAllOperation& operator=(WhenAllOperation&&) = delete;

            void start() & noexcept
            {
                if constexpr(sizeof...(T_Senders) == 0u)
                    m_receiver.set_value();
                else
                    (Holder<T_Index>::start(), ...);
            }

            decltype(auto) getEnv() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                requires requires(T_Receiver const& receiver) { receiver.get_env(); }
            {
                return m_receiver.get_env();
            }

            template<std::size_t T_I, typename... T>
            void setValue(T&&... values) noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    try
                    {
                        std::get<T_I>(m_values).emplace(std::forward<T>(values)...);
                    }
                    catch(...)
                    {
                        if(!m_error)
                            m_error = std::current_exception();
                    }
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

            void setError(std::exception_ptr error) noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    if(!m_error)
                        m_error = std::move(error);
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

            void setStopped() noexcept
            {
                bool complete;
                {
                    std::lock_guard lock(m_mutex);
                    m_stopped = true;
                    complete = --m_remaining == 0u;
                }
                if(complete)
                    finish();
            }

        private:
            void finish() noexcept
            {
                if(m_error)
                {
                    m_receiver.set_error(std::move(m_error));
                    return;
                }
                if(m_stopped)
                {
                    m_receiver.set_stopped();
                    return;
                }

                try
                {
                    auto values
                        = std::apply([](auto&... value) { return std::tuple_cat(std::move(*value)...); }, m_values);
                    std::apply(
                        [this](auto&&... value) { m_receiver.set_value(std::forward<decltype(value)>(value)...); },
                        std::move(values));
                }
                catch(...)
                {
                    m_receiver.set_error(std::current_exception());
                }
            }

            T_Receiver m_receiver;
            std::mutex m_mutex;
            std::size_t m_remaining = sizeof...(T_Senders);
            std::tuple<std::optional<ValueTupleOf<T_Senders>>...> m_values;
            std::exception_ptr m_error;
            bool m_stopped = false;
        };
    } // namespace detail

    template<typename... T_Senders>
    class WhenAllSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<
            typename detail::ValueSignatureFromTuple<detail::CombinedValueTuple<T_Senders...>>::type>;

        explicit WhenAllSender(T_Senders... senders) : m_senders(std::move(senders)...)
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::
                WhenAllOperation<std::decay_t<T_Receiver>, std::index_sequence_for<T_Senders...>, T_Senders...>{
                    std::move(m_senders),
                    std::forward<T_Receiver>(receiver)};
        }

    private:
        std::tuple<T_Senders...> m_senders;
    };

    template<typename... T_Senders>
    requires(Sender<T_Senders> && ...)
    auto whenAll(T_Senders... senders)
    {
        return WhenAllSender<T_Senders...>{std::move(senders)...};
    }
} // namespace caravan
