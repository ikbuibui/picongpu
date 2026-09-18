/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <caravan/alpaka/queue/managed.hpp>

namespace caravan::alpaka
{
    /** Structured lifetime for a runtime-sized set of split-phase producers.
     *
     * During serialized setup, add one ticket per producer and pass it to startSubmission(). Wrap the producer's
     * complete publication chain in ticket.publish(). waitFor() imports the published native dependencies; join()
     * completes only after its input AND every submitted producer are quiescent, including on errors. Tickets for
     * producers that never submit are failed when publication or the joined input fails.
     *
     * All producers must belong to the one joined input. Setup/connection happens before any producer starts.
     * Retirement notification is direct and allocation-free: there is no post-submission Event subscription.
     */
    template<typename T_Queue>
    class SubmissionGroup
    {
        using Work = SubmittedWork<T_Queue>;

        struct Slot
        {
            EventSource completion;
            std::optional<Work> work;
            bool submitted = false;
            bool retired = false;
        };

        struct State
        {
            std::mutex mutex;
            std::vector<Slot> slots;
            std::size_t remaining = 1u; // The joined input itself.
            std::exception_ptr error;
            std::exception_ptr missing
                = std::make_exception_ptr(std::logic_error("SubmissionGroup producer did not publish work"));
            void* receiver = nullptr;
            void (*deliver)(void*, std::exception_ptr) noexcept = nullptr;

            void rememberError(std::exception_ptr failure) noexcept
            {
                std::lock_guard lock(mutex);
                if(!error)
                    error = std::move(failure);
            }

            void arrive(std::exception_ptr failure) noexcept
            {
                std::exception_ptr result;
                {
                    std::lock_guard lock(mutex);
                    if(!error)
                        error = std::move(failure);
                    if(--remaining != 0u)
                        return;
                    result = error;
                }
                // Delivery can destroy the joined operation; callers retain State while notifying it.
                deliver(receiver, std::move(result));
            }

            void retire(std::size_t index, std::exception_ptr failure, bool onlyUnsubmitted = false) noexcept
            {
                {
                    std::lock_guard lock(mutex);
                    auto& slot = slots[index];
                    if(slot.retired || (onlyUnsubmitted && slot.submitted))
                        return;
                    slot.retired = true;
                }
                // Do not hold the group lock while waking resource-reuse continuations.
                if(failure)
                    slots[index].completion.setFailed(failure);
                else
                    slots[index].completion.setReady();
                arrive(std::move(failure));
            }

            void finishInput(std::exception_ptr failure) noexcept
            {
                // A producer that never published is a caller error and must fail the join.
                rememberError(failure ? failure : unpublishedFailure());
                for(std::size_t i = 0u; i < slots.size(); ++i)
                    retire(i, failure ? failure : missing, true);
                arrive({});
            }

            std::exception_ptr unpublishedFailure() noexcept
            {
                std::lock_guard lock(mutex);
                for(auto const& slot : slots)
                    if(!slot.work)
                        return missing;
                return {};
            }

            void waitOn(T_Queue& queue)
            {
                std::lock_guard lock(mutex);
                if(error)
                    std::rethrow_exception(error);
                for(auto const& slot : slots)
                {
                    if(!slot.work)
                        std::rethrow_exception(missing);
                    slot.work->waitOn(queue);
                }
            }
        };

        template<typename T_Sender, typename T_Receiver>
        class JoinOperation
        {
            struct Receiver
            {
                template<typename... T>
                void set_value(T&&...) noexcept
                {
                    auto state = owner->m_state;
                    state->finishInput({});
                }

                void set_error(std::exception_ptr error) noexcept
                {
                    auto state = owner->m_state;
                    state->finishInput(std::move(error));
                }

                decltype(auto) get_env() const noexcept(noexcept(std::declval<T_Receiver const&>().get_env()))
                    requires requires(T_Receiver const& receiver) { receiver.get_env(); }
                {
                    return std::as_const(owner->m_receiver).get_env();
                }

                JoinOperation* owner;
            };

        public:
            JoinOperation(std::shared_ptr<State> state, T_Sender sender, T_Receiver receiver)
                : m_state(std::move(state))
                , m_receiver(std::move(receiver))
                , m_operation(std::move(sender).connect(Receiver{this}))
            {
                if(m_state->deliver)
                    throw std::logic_error("SubmissionGroup can only be joined once");
                m_state->receiver = this;
                m_state->deliver = [](void* pointer, std::exception_ptr error) noexcept
                {
                    auto& receiver = static_cast<JoinOperation*>(pointer)->m_receiver;
                    if(error)
                        receiver.set_error(std::move(error));
                    else
                        receiver.set_value();
                };
            }

            JoinOperation(JoinOperation const&) = delete;
            JoinOperation& operator=(JoinOperation const&) = delete;
            JoinOperation(JoinOperation&&) = delete;
            JoinOperation& operator=(JoinOperation&&) = delete;

            void start() & noexcept
            {
                m_operation.start();
            }

        private:
            std::shared_ptr<State> m_state;
            T_Receiver m_receiver;
            decltype(std::declval<T_Sender&&>().connect(std::declval<Receiver>())) m_operation;
        };

        template<typename T_Sender>
        class JoinSender
        {
        public:
            using completion_signatures = CompletionSignatures<ValueSignature<>, ErrorSignature<std::exception_ptr>>;

            JoinSender(std::shared_ptr<State> state, T_Sender sender)
                : m_state(std::move(state))
                , m_sender(std::move(sender))
            {
            }

            template<typename T_Receiver>
            auto connect(T_Receiver&& receiver) &&
            {
                return JoinOperation<T_Sender, std::decay_t<T_Receiver>>{
                    std::move(m_state), std::move(m_sender), std::forward<T_Receiver>(receiver)};
            }

        private:
            std::shared_ptr<State> m_state;
            T_Sender m_sender;
        };

    public:
        /** Completion sink installed in a producer before submission, also used for resource-reuse ordering. */
        class Ticket
        {
            struct IdentityWork
            {
                Work operator()(Work work) const noexcept
                {
                    return work;
                }
            };

        public:
            Event event() const
            {
                return m_state->slots[m_index].completion.event();
            }

            void submitted() const noexcept
            {
                std::lock_guard lock(m_state->mutex);
                m_state->slots[m_index].submitted = true;
            }

            void setReady() const noexcept
            {
                auto state = m_state;
                state->retire(m_index, {});
            }

            void setFailed(std::exception_ptr error) const noexcept
            {
                auto state = m_state;
                state->retire(m_index, std::move(error));
            }

            /** Consume a startSubmission() result into this group. The extractor maps the value to SubmittedWork. */
            template<Sender T_Sender, typename T_Extract = IdentityWork>
            auto publish(T_Sender sender, T_Extract extract = {}) const
            {
                return materialize(std::move(sender))
                       | then(
                           [ticket = *this, extract = std::move(extract)](auto result) mutable
                           {
                               auto state = ticket.m_state;
                               if(result.hasValue())
                               {
                                   auto values = result.takeValues();
                                   auto work = std::invoke(extract, std::move(std::get<0>(values)));
                                   std::lock_guard lock(state->mutex);
                                   state->slots[ticket.m_index].work.emplace(std::move(work));
                               }
                               else
                               {
                                   state->rememberError(result.error());
                                   // A failure after submission must still wait for the producer's notification.
                                   state->retire(ticket.m_index, result.error(), true);
                               }
                           });
            }

        private:
            friend class SubmissionGroup;

            Ticket(std::shared_ptr<State> state, std::size_t index) : m_state(std::move(state)), m_index(index)
            {
            }

            std::shared_ptr<State> m_state;
            std::size_t m_index;
        };

        SubmissionGroup() : m_state(std::make_shared<State>())
        {
        }

        Ticket add()
        {
            if(m_state->deliver)
                throw std::logic_error("SubmissionGroup setup must precede connection");
            auto index = m_state->slots.size();
            m_state->slots.emplace_back();
            ++m_state->remaining;
            return Ticket{m_state, index};
        }

        auto waitFor() const
        {
            return submit([state = m_state](T_Queue& queue) { state->waitOn(queue); });
        }

        template<Sender T_Sender>
        auto join(T_Sender sender) const
        {
            return JoinSender<T_Sender>{m_state, std::move(sender)};
        }

    private:
        std::shared_ptr<State> m_state;
    };
} // namespace caravan::alpaka
