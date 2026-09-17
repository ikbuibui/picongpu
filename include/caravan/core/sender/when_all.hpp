/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    namespace detail
    {
        template<std::size_t T_Index, typename T_Owner, typename T_Environment>
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

            decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Environment const&>().get_env()))
                requires requires(T_Environment const& value) { value.get_env(); }
            {
                return environment->get_env();
            }

            T_Owner* owner;
            T_Environment const* environment;
        };

        template<typename T_Owner, typename T_Environment>
        struct RuntimeWhenAllReceiver
        {
            template<typename... T>
            void set_value(T&&...) noexcept
            {
                owner->complete({});
            }

            void set_error(std::exception_ptr error) noexcept
            {
                owner->complete(std::move(error));
            }

            decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Environment const&>().get_env()))
                requires requires(T_Environment const& value) { value.get_env(); }
            {
                return environment->get_env();
            }

            T_Owner* owner;
            T_Environment const* environment;
        };

        template<typename T_Sender, typename T_Receiver>
        class RuntimeWhenAllOperation
        {
            using Self = RuntimeWhenAllOperation;
            using Receiver = RuntimeWhenAllReceiver<Self, T_Receiver>;
            using ChildOperation = decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>()));

            struct Child
            {
                Child(T_Sender sender, Receiver receiver)
                    : operation(std::move(sender).connect(std::move(receiver)))
                {
                }

                ChildOperation operation;
            };

        public:
            RuntimeWhenAllOperation(std::vector<T_Sender> senders, T_Receiver receiver)
                : m_receiver(std::move(receiver))
                , m_remaining(senders.size())
            {
                m_operations.reserve(senders.size());
                for(auto& sender : senders)
                    m_operations.emplace_back(
                        std::make_unique<Child>(std::move(sender), Receiver{this, &m_receiver}));
            }

            RuntimeWhenAllOperation(RuntimeWhenAllOperation const&) = delete;
            RuntimeWhenAllOperation& operator=(RuntimeWhenAllOperation const&) = delete;
            RuntimeWhenAllOperation(RuntimeWhenAllOperation&&) = delete;
            RuntimeWhenAllOperation& operator=(RuntimeWhenAllOperation&&) = delete;

            void start() & noexcept
            {
                if(m_operations.empty())
                {
                    m_receiver.set_value();
                    return;
                }

                // The final synchronous completion may destroy this operation.
                for(std::size_t i = 0u; i + 1u < m_operations.size(); ++i)
                    m_operations[i]->operation.start();
                m_operations.back()->operation.start();
            }

            void complete(std::exception_ptr error) noexcept
            {
                bool finished;
                std::exception_ptr finalError;
                {
                    std::lock_guard lock(m_mutex);
                    if(error && !m_error)
                        m_error = std::move(error);
                    finished = --m_remaining == 0u;
                    if(finished)
                        finalError = m_error;
                }
                if(!finished)
                    return;
                if(finalError)
                    m_receiver.set_error(std::move(finalError));
                else
                    m_receiver.set_value();
            }

        private:
            T_Receiver m_receiver;
            std::vector<std::unique_ptr<Child>> m_operations;
            std::mutex m_mutex;
            std::size_t m_remaining;
            std::exception_ptr m_error;
        };

        template<typename T_Receiver>
        class WhenAllReceiverHolder
        {
        protected:
            explicit WhenAllReceiverHolder(T_Receiver receiver) : m_receiver(std::move(receiver))
            {
            }

            T_Receiver m_receiver;
        };

        template<std::size_t T_Index, typename T_Owner, typename T_Environment, typename T_Sender>
        class WhenAllOperationHolder
        {
            using Receiver = WhenAllReceiver<T_Index, T_Owner, T_Environment>;

        public:
            WhenAllOperationHolder(T_Sender sender, T_Owner* owner, T_Environment const* environment)
                : m_operation(std::move(sender).connect(Receiver{owner, environment}))
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
            : private WhenAllReceiverHolder<T_Receiver>
            , private WhenAllOperationHolder<
                  T_Index,
                  WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>,
                  T_Receiver,
                  T_Senders>...
        {
            using Self = WhenAllOperation<T_Receiver, std::index_sequence<T_Index...>, T_Senders...>;
            using ReceiverHolder = WhenAllReceiverHolder<T_Receiver>;

            template<std::size_t T_I>
            using Holder
                = WhenAllOperationHolder<T_I, Self, T_Receiver, std::tuple_element_t<T_I, std::tuple<T_Senders...>>>;

        public:
            WhenAllOperation(std::tuple<T_Senders...> senders, T_Receiver receiver)
                : ReceiverHolder(std::move(receiver))
                , Holder<T_Index>(std::move(std::get<T_Index>(senders)), this, &this->m_receiver)...
            {
            }

            WhenAllOperation(WhenAllOperation const&) = delete;
            WhenAllOperation& operator=(WhenAllOperation const&) = delete;
            WhenAllOperation(WhenAllOperation&&) = delete;
            WhenAllOperation& operator=(WhenAllOperation&&) = delete;

            void start() & noexcept
            {
                if constexpr(sizeof...(T_Senders) == 0u)
                    this->m_receiver.set_value();
                else
                    (Holder<T_Index>::start(), ...);
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

        private:
            void finish() noexcept
            {
                if(m_error)
                {
                    this->m_receiver.set_error(std::move(m_error));
                    return;
                }

                try
                {
                    auto values
                        = std::apply([](auto&... value) { return std::tuple_cat(std::move(*value)...); }, m_values);
                    std::apply(
                        [this](auto&&... value)
                        { this->m_receiver.set_value(std::forward<decltype(value)>(value)...); },
                        std::move(values));
                }
                catch(...)
                {
                    this->m_receiver.set_error(std::current_exception());
                }
            }

            std::mutex m_mutex;
            std::size_t m_remaining = sizeof...(T_Senders);
            std::tuple<std::optional<ValueTupleOf<T_Senders>>...> m_values;
            std::exception_ptr m_error;
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

    /** Join a runtime-sized homogeneous set of senders, discarding their values. */
    template<Sender T_Sender>
    class RuntimeWhenAllSender
    {
    public:
        using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

        explicit RuntimeWhenAllSender(std::vector<T_Sender> senders) : m_senders(std::move(senders))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::RuntimeWhenAllOperation<T_Sender, std::decay_t<T_Receiver>>{
                std::move(m_senders),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        std::vector<T_Sender> m_senders;
    };

    template<Sender T_Sender>
    auto whenAll(std::vector<T_Sender> senders)
    {
        return RuntimeWhenAllSender<T_Sender>{std::move(senders)};
    }

    /** Join senders without domain-specific fusion. */
    template<Sender... T_Senders>
    auto genericWhenAll(T_Senders... senders)
    {
        return WhenAllSender<T_Senders...>{std::move(senders)...};
    }

    template<typename... T_Senders>
    requires(Sender<T_Senders> && ...)
    auto whenAll(T_Senders... senders)
    {
        auto domain = detail::commonDomain(senders...);
        if constexpr(requires { domain.transform(WhenAllTag{}, std::move(senders)...); })
            return domain.transform(WhenAllTag{}, std::move(senders)...);
        else
            return genericWhenAll(std::move(senders)...);
    }
} // namespace caravan
