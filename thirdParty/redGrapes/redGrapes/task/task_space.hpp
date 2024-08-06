/* Copyright 2021-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/memory/block.hpp"
#include "redGrapes/resource/resource_user.hpp"
#include "redGrapes/task/property/id.hpp"
#include "redGrapes/util/trace.hpp"

#include <atomic>
#include <memory>

namespace redGrapes
{

    /*! TaskSpace handles sub-taskspaces of child tasks
     */
    struct TaskSpace : std::enable_shared_from_this<TaskSpace>
    {
        std::atomic<TaskID> task_count;
        unsigned depth;
        ResourceUser* parent;

        // top space
        TaskSpace() : task_count(0), depth(0), parent(nullptr)
        {
        }

        // sub space
        template<typename TTask>
        requires std::is_base_of_v<ResourceUser, TTask>
        TaskSpace(TTask* parent) : task_count(0)
                                 , depth(parent->space->depth + 1)
                                 , parent(parent)
        {
        }

        // add a new task to the task-space
        template<typename TTask>
        requires std::is_base_of_v<ResourceUser, TTask>
        void submit(TTask* task)
        {
            TRACE_EVENT("TaskSpace", "submit()");
            task->space = this->shared_from_this();
            task->task = task;

            ++task_count;

            if(parent)
            {
                assert(parent->is_superset_of(*task));
                // add dependency to parent
                SPDLOG_TRACE("add event dep to parent");
                task->post_event.add_follower(static_cast<TTask*>(parent)->get_post_event());
            }

            for(auto r = task->unique_resources.rbegin(); r != task->unique_resources.rend(); ++r)
            {
                r->user_entry = r->resource->users.push(task);
            }
        }

        // remove task from task-space
        template<typename TTask>
        requires std::is_base_of_v<ResourceUser, TTask> && std::is_base_of_v<ResourceUser, TTask>
        void free_task(TTask* task)
        {
            TRACE_EVENT("TaskSpace", "free_task()");
            TaskID count = task_count.fetch_sub(1) - 1;

            WorkerId worker_id = task->worker_id;
            task->~TTask();

            // FIXME: len of the Block is not correct since FunTask object is bigger than sizeof(Task)
            TaskFreeCtx::worker_alloc_pool.get_alloc(worker_id).deallocate(
                memory::Block{(uintptr_t) task, sizeof(TTask)});

            // TODO: implement this using post-event of root-task?
            //  - event already has in_edge count
            //  -> never have current_task = nullptr
            // SPDLOG_INFO("kill task... {} remaining", count);

            // to wake after barrier the check tells us that the root space is empty
            // if(all_tasks_done())
            if(count == 0 && depth == 0)
            {
                SPDLOG_TRACE("Wake up parser due to free task and no more tasks");
                // TODO should i wake up workers also? that was the old behaviour
                TaskFreeCtx::cv.notify();
            }
        }

        bool empty() const
        {
            TaskID tc = task_count.load();
            return tc == 0;
        }
    };

} // namespace redGrapes
