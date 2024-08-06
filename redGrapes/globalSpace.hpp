/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/resource/resource_user.hpp"
#include "redGrapes/scheduler/event.hpp"
#include "redGrapes/task/task_space.hpp"

#include <spdlog/spdlog.h>

#include <optional>

namespace redGrapes
{

    // not a Task because Task type is a template, so we static cast current_task to Task type everywhere
    inline thread_local ResourceUser* current_task;

    inline std::shared_ptr<TaskSpace> root_space;

    inline std::shared_ptr<TaskSpace> current_task_space()
    {
        if(current_task)
        {
            if(!current_task->children)
            {
                auto task_space = std::make_shared<TaskSpace>(current_task);
                SPDLOG_TRACE("create child space = {}", (void*) task_space.get());
                current_task->children = task_space;
            }

            return current_task->children;
        }
        else
            return root_space;
    }

    inline unsigned scope_depth_impl()
    {
        if(auto ts = current_task_space())
            return ts->depth;
        else
            return 0;
    }

    //! pause the currently running task at least until event is reached
    // else is supposed to be called when .get()/.submit() is called on emplace task, which calls the future
    // .get(),
    // if there is no current task at that time (not in a child task space) in the root space,wake up the parser
    // thread
    // we can assert(event.task != 0);
    template<typename TTask>
    void yield_impl(scheduler::EventPtr<TTask> event)
    {
        if(current_task)
        {
            while(!event->is_reached())
                static_cast<TTask*>(current_task)->yield(event);
        }
        else
        {
            event->waker_id = parserID;
            while(!event->is_reached())
                TaskFreeCtx::idle();
        }
    }

    /*! Create an event on which the termination of the current task depends.
     *  A task must currently be running.
     *
     * @return Handle to flag the event with `reach_event` later.
     *         nullopt if there is no task running currently
     */
    template<typename TTask>
    std::optional<scheduler::EventPtr<TTask>> create_event_impl()
    {
        if(current_task)
            return static_cast<TTask*>(current_task)->make_event();
        else
            return std::nullopt;
    }

} // namespace redGrapes
