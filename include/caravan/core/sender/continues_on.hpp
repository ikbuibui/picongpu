/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
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
    namespace detail
    {
        template<typename T_Sender, typename T_Scheduler, typename T_Receiver>
        class ContinuesOnOperation
        {
            struct TransferReceiver
            {
                template<typename... T>
                void set_value(T&&... values) noexcept
                {
                    owner->transferValue(std::forward<T>(values)...);
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->transferError(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->transferStopped();
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                ContinuesOnOperation* owner;
            };

            struct ScheduleReceiver
            {
                void set_value() noexcept
                {
                    owner->complete();
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    owner->m_receiver.set_error(std::move(error));
                }

                void set_stopped() noexcept
                {
                    owner->m_receiver.set_stopped();
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return owner->m_receiver.get_env();
                }

                ContinuesOnOperation* owner;
            };

        public:
            ContinuesOnOperation(T_Sender sender, T_Scheduler scheduler, T_Receiver receiver)
                : m_receiver(std::move(receiver))
                , m_scheduled(scheduler.schedule().connect(ScheduleReceiver{this}))
                , m_upstream(std::move(sender).connect(TransferReceiver{this}))
            {
            }

            ContinuesOnOperation(ContinuesOnOperation const&) = delete;
            ContinuesOnOperation& operator=(ContinuesOnOperation const&) = delete;
            ContinuesOnOperation(ContinuesOnOperation&&) = delete;
            ContinuesOnOperation& operator=(ContinuesOnOperation&&) = delete;

            void start() & noexcept
            {
                m_upstream.start();
            }

        private:
            template<typename... T>
            void transferValue(T&&... values) noexcept
            {
                try
                {
                    m_values.emplace(std::forward<T>(values)...);
                }
                catch(...)
                {
                    transferError(std::current_exception());
                    return;
                }
                m_scheduled.start();
            }

            void transferError(std::exception_ptr error) noexcept
            {
                m_error = std::move(error);
                m_scheduled.start();
            }

            void transferStopped() noexcept
            {
                m_stopped = true;
                m_scheduled.start();
            }

            void complete() noexcept
            {
                if(m_values)
                    std::apply(
                        [this](auto&&... values) { m_receiver.set_value(std::forward<decltype(values)>(values)...); },
                        std::move(*m_values));
                else if(m_stopped)
                    m_receiver.set_stopped();
                else
                    m_receiver.set_error(std::move(m_error));
            }

            T_Receiver m_receiver;
            std::optional<StoredValueTuple<T_Sender>> m_values;
            std::exception_ptr m_error;
            bool m_stopped = false;
            decltype(std::declval<T_Scheduler&>().schedule().connect(std::declval<ScheduleReceiver>())) m_scheduled;
            decltype(std::declval<T_Sender&&>().connect(std::declval<TransferReceiver>())) m_upstream;
        };
    } // namespace detail

    template<typename T_Sender, typename T_Scheduler>
    class ContinuesOnSender
    {
    public:
        using completion_signatures = CompletionSignaturesOf<T_Sender>;

        ContinuesOnSender(T_Sender sender, T_Scheduler scheduler)
            : m_sender(std::move(sender))
            , m_scheduler(std::move(scheduler))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::ContinuesOnOperation<T_Sender, T_Scheduler, std::decay_t<T_Receiver>>{
                std::move(m_sender),
                std::move(m_scheduler),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Sender m_sender;
        T_Scheduler m_scheduler;
    };

    /** Transfer every upstream completion channel onto an explicit scheduler.
     *
     * Values remain owned by the connected operation until delivery. If scheduling
     * fails or stops, that completion replaces the stored upstream completion.
     */
    template<Sender T_Sender, typename T_Scheduler>
    auto continuesOn(T_Sender sender, T_Scheduler scheduler)
    {
        return ContinuesOnSender<T_Sender, T_Scheduler>{std::move(sender), std::move(scheduler)};
    }

    template<typename T_Scheduler>
    auto continuesOn(T_Scheduler scheduler)
    {
        return detail::SenderAdaptorClosure{[scheduler = std::move(scheduler)](auto sender) mutable
                                            { return continuesOn(std::move(sender), std::move(scheduler)); }};
    }
} // namespace caravan
