/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

#include <caravan/core/sender.hpp>

namespace caravan::alpaka
{
    namespace detail
    {
        template<typename T_Queue, typename T_Submit, typename T_Receiver>
        class SubmitOperation
        {
        public:
            SubmitOperation(T_Queue& queue, T_Submit submit, T_Receiver receiver)
                : m_queue(&queue)
                , m_submit(std::move(submit))
                , m_receiver(std::move(receiver))
            {
            }

            SubmitOperation(SubmitOperation const&) = delete;
            SubmitOperation& operator=(SubmitOperation const&) = delete;
            SubmitOperation(SubmitOperation&&) = delete;
            SubmitOperation& operator=(SubmitOperation&&) = delete;

            void start() & noexcept
            {
                try
                {
                    std::invoke(m_submit, *m_queue);
                    ::alpaka::enqueue(*m_queue, [this]() noexcept { m_receiver.set_value(); });
                }
                catch(...)
                {
                    auto error = std::current_exception();
                    try
                    {
                        // Submission may have partially succeeded. Keep captures alive until the queue is safe.
                        ::alpaka::wait(*m_queue);
                    }
                    catch(...)
                    {
                        error = std::current_exception();
                    }
                    m_receiver.set_error(std::move(error));
                }
            }

        private:
            T_Queue* m_queue;
            T_Submit m_submit;
            T_Receiver m_receiver;
        };
    } // namespace detail

    /** Lazily submit one same-queue batch and complete after its native work.
     *
     * The queue is borrowed and must outlive the operation. The submit callable is
     * retained in operation state, receives the queue on start, and may enqueue a
     * kernel/copy/fill chain. Queue FIFO preserves that chain without a host wait;
     * the final alpaka host callback publishes host-visible completion.
     *
     * ponytail: one queue batch only; add native event/domain chaining when a real
     * cross-queue consumer requires it.
     */
    template<typename T_Queue, typename T_Submit>
    class SubmitSender
    {
    public:
        using completion_signatures
            = CompletionSignatures<ValueSignature<>, ErrorSignature<std::exception_ptr>, StoppedSignature>;

        SubmitSender(T_Queue& queue, T_Submit submit) : m_queue(&queue), m_submit(std::move(submit))
        {
        }

        template<typename T_Receiver>
        auto connect(T_Receiver&& receiver) &&
        {
            return detail::SubmitOperation<T_Queue, T_Submit, std::decay_t<T_Receiver>>{
                *m_queue,
                std::move(m_submit),
                std::forward<T_Receiver>(receiver)};
        }

    private:
        T_Queue* m_queue;
        T_Submit m_submit;
    };

    template<typename T_Queue, typename T_Submit>
    auto submit(T_Queue& queue, T_Submit submit)
    {
        static_assert(::alpaka::isQueue<T_Queue>);
        return SubmitSender<T_Queue, T_Submit>{queue, std::move(submit)};
    }
} // namespace caravan::alpaka
