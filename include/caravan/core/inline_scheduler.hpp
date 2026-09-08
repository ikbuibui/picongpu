/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <type_traits>
#include <utility>

#include <caravan/core/sender.hpp>

namespace caravan
{
    /** Scheduler that completes work immediately on the calling thread. */
    class InlineScheduler
    {
        class ScheduleSender
        {
        public:
            using completion_signatures = detail::DefaultCompletionSignatures<ValueSignature<>>;

            template<typename T_Receiver>
            class Operation
            {
            public:
                explicit Operation(T_Receiver receiver) : m_receiver(std::move(receiver))
                {
                }

                void start() & noexcept
                {
                    m_receiver.set_value();
                }

            private:
                T_Receiver m_receiver;
            };

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return Operation<std::decay_t<T_Receiver>>{std::forward<T_Receiver>(receiver)};
            }
        };

    public:
        template<typename T_Function>
        void post(T_Function&& function) const
        {
            std::forward<T_Function>(function)();
        }

        auto schedule() const noexcept
        {
            return ScheduleSender{};
        }
    };

} // namespace caravan
