/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <concepts>
#include <exception>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace caravan
{
    template<typename... T>
    struct ValueSignature
    {
    };

    template<typename T>
    struct ErrorSignature
    {
    };

    struct StoppedSignature
    {
    };

    template<typename... T_Signatures>
    struct CompletionSignatures
    {
    };

    template<typename T_Sender>
    using CompletionSignaturesOf = typename std::remove_cvref_t<T_Sender>::completion_signatures;

    template<typename T>
    inline constexpr bool isSupportedCompletionSignatures = false;

    template<typename... T>
    inline constexpr bool isSupportedCompletionSignatures<
        CompletionSignatures<ValueSignature<T...>, ErrorSignature<std::exception_ptr>, StoppedSignature>>
        = ((!std::is_reference_v<T> && std::is_same_v<T, std::remove_cv_t<T>>) && ...);

    /** The deliberately narrow Caravan migration-sender profile.
     *
     * A sender has exactly one alternative of owned values, an exception_ptr
     * error channel, and a stopped channel. Stopped completion is propagated but
     * does not imply cancellation support. Receiver environments are forwarded
     * by composition; scoped placement queries the logical current scheduler.
     */
    template<typename T_Sender>
    concept Sender = requires { typename CompletionSignaturesOf<T_Sender>; }
                     && isSupportedCompletionSignatures<CompletionSignaturesOf<T_Sender>>;

    template<typename T_Operation>
    concept OperationState = requires(T_Operation& operation) {
        { operation.start() } noexcept;
    };

    template<typename T_Sender, typename T_Receiver>
    concept SenderTo = Sender<T_Sender> && requires(T_Sender&& sender, T_Receiver&& receiver) {
        { std::forward<T_Sender>(sender).connect(std::forward<T_Receiver>(receiver)) } -> OperationState;
    };

    namespace detail
    {
        template<typename T_Signatures>
        struct ValueTuple;

        template<typename... T>
        struct ValueTuple<
            CompletionSignatures<ValueSignature<T...>, ErrorSignature<std::exception_ptr>, StoppedSignature>>
        {
            using type = std::tuple<T...>;
        };

        template<typename T_Sender>
        using ValueTupleOf = typename ValueTuple<CompletionSignaturesOf<T_Sender>>::type;

        template<typename T_Tuple>
        struct DecayedTuple;

        template<typename... T>
        struct DecayedTuple<std::tuple<T...>>
        {
            using type = std::tuple<std::decay_t<T>...>;
        };

        template<typename T_Tuple>
        using DecayedTupleOf = typename DecayedTuple<T_Tuple>::type;

        template<typename T_Tuple>
        struct LvalueTuple;

        template<typename... T>
        struct LvalueTuple<std::tuple<T...>>
        {
            using type = std::tuple<T&...>;
        };

        template<typename T_Tuple>
        struct ValueSignatureFromTuple;

        template<typename... T>
        struct ValueSignatureFromTuple<std::tuple<T...>>
        {
            using type = ValueSignature<T...>;
        };

        template<typename T_Function, typename T_Tuple>
        struct InvokeResultFromTuple;

        template<typename T_Function, typename... T>
        struct InvokeResultFromTuple<T_Function, std::tuple<T...>>
        {
            using type = std::invoke_result_t<T_Function&, T...>;
        };

        template<typename T_Result>
        using ResultValueSignature = std::
            conditional_t<std::is_void_v<T_Result>, ValueSignature<>, ValueSignature<std::remove_cvref_t<T_Result>>>;

        template<typename T_ValueSignature>
        using DefaultCompletionSignatures
            = CompletionSignatures<T_ValueSignature, ErrorSignature<std::exception_ptr>, StoppedSignature>;

        template<typename T_Sender, typename T_Function>
        using ThenCompletionSignatures = DefaultCompletionSignatures<
            ResultValueSignature<typename InvokeResultFromTuple<T_Function, ValueTupleOf<T_Sender>>::type>>;

        template<typename T_Sender>
        using StoredValueTuple = DecayedTupleOf<ValueTupleOf<T_Sender>>;

        template<typename T_Sender, typename T_Function>
        using SuccessorSender =
            typename InvokeResultFromTuple<T_Function, typename LvalueTuple<StoredValueTuple<T_Sender>>::type>::type;

        template<typename... T_Senders>
        using CombinedValueTuple = decltype(std::tuple_cat(std::declval<ValueTupleOf<T_Senders>>()...));

        template<typename T_Function>
        class SenderAdaptorClosure
        {
        public:
            explicit SenderAdaptorClosure(T_Function function) : m_function(std::move(function))
            {
            }

            template<Sender T_Sender>
            friend auto operator|(T_Sender sender, SenderAdaptorClosure closure)
            {
                return std::invoke(std::move(closure.m_function), std::move(sender));
            }

        private:
            T_Function m_function;
        };
    } // namespace detail
} // namespace caravan
