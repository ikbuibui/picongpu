/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/core/sender/common.hpp>

namespace caravan
{
    /** The terminal value or error of a materialized sender. */
    template<typename... T>
    class CompletionResult
    {
    public:
        template<typename... U>
        explicit CompletionResult(std::in_place_t, U&&... values)
            : m_values(std::in_place, std::forward<U>(values)...)
        {
        }

        explicit CompletionResult(std::exception_ptr error) : m_error(std::move(error))
        {
        }

        bool hasValue() const noexcept
        {
            return m_values.has_value();
        }

        std::exception_ptr const& error() const noexcept
        {
            return m_error;
        }

        std::tuple<T...> takeValues()
        {
            return std::move(*m_values);
        }

    private:
        std::optional<std::tuple<T...>> m_values;
        std::exception_ptr m_error;
    };

    namespace detail
    {
        template<typename T_Tuple>
        struct CompletionResultFromTuple;

        template<typename... T>
        struct CompletionResultFromTuple<std::tuple<T...>>
        {
            using type = CompletionResult<T...>;
        };

        template<typename T_Sender>
        using CompletionResultOf = typename CompletionResultFromTuple<ValueTupleOf<T_Sender>>::type;

        template<typename T_Receiver, typename T_Result>
        struct MaterializeReceiver
        {
            template<typename... T>
            void set_value(T&&... values) noexcept
            {
                try
                {
                    receiver->set_value(T_Result{std::in_place, std::forward<T>(values)...});
                }
                catch(...)
                {
                    receiver->set_error(std::current_exception());
                }
            }

            void set_error(std::exception_ptr error) noexcept
            {
                receiver->set_value(T_Result{std::move(error)});
            }

            decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                requires requires(T_Receiver const& output) { output.get_env(); }
            {
                return std::as_const(*receiver).get_env();
            }

            T_Receiver* receiver;
        };

        template<typename T_Sender, typename T_Receiver>
        class MaterializeOperation
        {
            using Result = CompletionResultOf<T_Sender>;
            using Receiver = MaterializeReceiver<T_Receiver, Result>;

        public:
            MaterializeOperation(T_Sender sender, T_Receiver receiver)
                : m_receiver(std::move(receiver))
                , m_operation(std::move(sender).connect(Receiver{&m_receiver}))
            {
            }

            MaterializeOperation(MaterializeOperation const&) = delete;
            MaterializeOperation& operator=(MaterializeOperation const&) = delete;
            MaterializeOperation(MaterializeOperation&&) = delete;
            MaterializeOperation& operator=(MaterializeOperation&&) = delete;

            void start() & noexcept
            {
                m_operation.start();
            }

        private:
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };
    } // namespace detail

    /** Convert a sender's value/error channels into one value channel. */
    template<Sender T_Sender>
    class MaterializeSender
    {
        using Result = detail::CompletionResultOf<T_Sender>;

    public:
        using completion_signatures
            = CompletionSignatures<ValueSignature<Result>, ErrorSignature<std::exception_ptr>>;

        explicit MaterializeSender(T_Sender sender) : m_sender(std::move(sender))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::MaterializeOperation<T_Sender, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
    };

    template<Sender T_Sender>
    auto materialize(T_Sender sender)
    {
        return MaterializeSender<T_Sender>{std::move(sender)};
    }

    inline auto materialize()
    {
        return detail::SenderAdaptorClosure{[](auto sender) { return materialize(std::move(sender)); }};
    }
} // namespace caravan
