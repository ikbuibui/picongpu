/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <stdexcept>
#include <utility>

#include <caravan/core.hpp>

namespace pmacc::async
{
    /** Explicit owner and host-continuation context for PMacc asynchronous work. */
    class Context
    {
    public:
        Context() = default;
        Context(Context const&) = delete;
        Context& operator=(Context const&) = delete;

        ~Context()
        {
            wait(m_scope.join());
            m_loop.finish();
        }

        template<typename T_Sender>
        caravan::Event spawn(T_Sender sender)
        {
            return m_scope.spawn(caravan::continuesOn(std::move(sender), m_loop.scheduler()));
        }

        template<typename T, typename T_Sender>
        caravan::Future<T> spawnFuture(T_Sender sender)
        {
            return m_scope.spawnFuture<T>(caravan::continuesOn(std::move(sender), m_loop.scheduler()));
        }

        /** Drive host continuations until this operation is terminal. */
        void wait(caravan::Event const& event)
        {
            if(caravan::isExecutorThread() && event.state() == caravan::CompletionState::pending)
                throw std::logic_error("A PMacc async continuation cannot wait on pending work");
            while(event.state() == caravan::CompletionState::pending)
                m_loop.runOne();
            event.wait();
        }

        void runReady()
        {
            m_loop.runReady();
        }

        caravan::RunLoopScheduler scheduler() noexcept
        {
            return m_loop.scheduler();
        }

    private:
        caravan::RunLoop m_loop;
        caravan::AsyncScope m_scope;
    };
} // namespace pmacc::async
