/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <caravan/core/eager.hpp>

namespace caravan
{
    namespace detail
    {
        class SpawnOperation
        {
        public:
            virtual ~SpawnOperation() = default;
            virtual void start() noexcept = 0;
        };

        class AsyncScopeState
        {
        public:
            std::size_t reserve()
            {
                std::lock_guard lock(m_mutex);
                if(m_closed)
                    throw std::logic_error("Cannot spawn into a joined Caravan async scope");
                auto const id = m_nextId++;
                m_operations.emplace(id, nullptr);
                return id;
            }

            void attach(std::size_t id, std::shared_ptr<SpawnOperation> operation)
            {
                std::lock_guard lock(m_mutex);
                m_operations.at(id) = std::move(operation);
            }

            void complete(std::size_t id) noexcept
            {
                std::shared_ptr<SpawnOperation> operation;
                bool joined = false;
                {
                    std::lock_guard lock(m_mutex);
                    auto const found = m_operations.find(id);
                    if(found == m_operations.end())
                        return;
                    operation = std::move(found->second);
                    m_operations.erase(found);
                    joined = m_closed && m_operations.empty();
                }
                if(joined)
                    m_joined.setReady();
            }

            Event join()
            {
                bool joined = false;
                {
                    std::lock_guard lock(m_mutex);
                    m_closed = true;
                    joined = m_operations.empty();
                }
                if(joined)
                    m_joined.setReady();
                return m_joined.event();
            }

        private:
            std::mutex m_mutex;
            std::unordered_map<std::size_t, std::shared_ptr<SpawnOperation>> m_operations;
            std::size_t m_nextId = 0u;
            bool m_closed = false;
            EventSource m_joined;
        };

        struct ScopeReceiver
        {
            template<typename... T>
            void set_value(T&&...) noexcept
            {
                output.setReady();
                scope->complete(id);
            }

            void set_error(std::exception_ptr error) noexcept
            {
                output.setFailed(std::move(error));
                scope->complete(id);
            }

            void set_stopped() noexcept
            {
                output.setStopped();
                scope->complete(id);
            }

            std::shared_ptr<AsyncScopeState> scope;
            EventSource output;
            std::size_t id;
        };

        template<typename T>
        struct FutureScopeReceiver
        {
            template<typename U>
            void set_value(U&& value) noexcept
            {
                try
                {
                    output.setValue(std::forward<U>(value));
                }
                catch(...)
                {
                    output.setFailed(std::current_exception());
                }
                scope->complete(id);
            }

            void set_error(std::exception_ptr error) noexcept
            {
                output.setFailed(std::move(error));
                scope->complete(id);
            }

            void set_stopped() noexcept
            {
                output.setStopped();
                scope->complete(id);
            }

            std::shared_ptr<AsyncScopeState> scope;
            Promise<T> output;
            std::size_t id;
        };

        template<typename T_Sender, typename T_Receiver>
        class SpawnOperationModel final : public SpawnOperation
        {
        public:
            SpawnOperationModel(T_Sender sender, T_Receiver receiver)
                : m_operation(std::move(sender).connect(std::move(receiver)))
            {
            }

            void start() noexcept override
            {
                m_operation.start();
            }

        private:
            decltype(std::declval<T_Sender&&>().connect(std::declval<T_Receiver>())) m_operation;
        };
    } // namespace detail

    /** Owns eagerly spawned sender operations until receiver completion.
     *
     * join() closes the scope to new work and completes when every operation is
     * quiescent. The returned Event carries each sender's value/error/stopped
     * channel; values are deliberately type-erased at this migration boundary.
     */
    class AsyncScope
    {
    public:
        AsyncScope() : m_state(std::make_shared<detail::AsyncScopeState>())
        {
        }

        AsyncScope(AsyncScope const&) = delete;
        AsyncScope& operator=(AsyncScope const&) = delete;

        ~AsyncScope()
        {
            m_state->join().wait();
        }

        template<typename T_Sender>
        Event spawn(T_Sender sender)
        {
            EventSource output;
            auto result = output.event();
            auto const id = m_state->reserve();
            try
            {
                using Receiver = detail::ScopeReceiver;
                using Operation = detail::SpawnOperationModel<T_Sender, Receiver>;
                auto operation = std::make_shared<Operation>(std::move(sender), Receiver{m_state, output, id});
                m_state->attach(id, operation);
                operation->start();
            }
            catch(...)
            {
                m_state->complete(id);
                throw;
            }
            return result;
        }

        /** Eagerly spawn a single-value sender and retain its operation state. */
        template<typename T, typename T_Sender>
        Future<T> spawnFuture(T_Sender sender)
        {
            Promise<T> output;
            auto result = output.future();
            auto const id = m_state->reserve();
            try
            {
                using Receiver = detail::FutureScopeReceiver<T>;
                using Operation = detail::SpawnOperationModel<T_Sender, Receiver>;
                auto operation = std::make_shared<Operation>(std::move(sender), Receiver{m_state, output, id});
                m_state->attach(id, operation);
                operation->start();
            }
            catch(...)
            {
                m_state->complete(id);
                throw;
            }
            return result;
        }

        Event join()
        {
            return m_state->join();
        }

    private:
        std::shared_ptr<detail::AsyncScopeState> m_state;
    };
} // namespace caravan
