/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/memory/hwloc_alloc.hpp"

#include <optional>
#include <thread>

namespace redGrapes::dispatch::thread
{
    template<typename Worker>
    struct WorkerThread
    {
        std::thread thread;
        hwloc_obj_t const obj; // storing this vs calculating this as
        // hwloc_obj_t obj
        //     = hwloc_get_obj_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU, this->id % TaskFreeCtx::n_pus);
        Worker worker;

        template<typename... Args>
        WorkerThread(hwloc_obj_t const obj, Args&&... args) : obj{obj}
                                                            , worker{std::forward<Args>(args)...}
        {
        }

        ~WorkerThread()
        {
        }

        void start()
        {
            thread = std::thread([this] { this->run(); });
        }

        void stop()
        {
            worker.stop();
            thread.join();
        }

        /* function the thread will execute
         */
        void run()
        {
            /* setup membind- & cpubind policies using hwloc
             */
            this->cpubind();
            this->membind();

            /* initialize thread-local variables
             */
            *TaskFreeCtx::current_worker_id = worker.id;

            /* execute tasks until stop()
             */
            worker.work_loop();

            TaskFreeCtx::current_worker_id = std::nullopt;

            SPDLOG_TRACE("Worker Finished!");
        }

        void cpubind()
        {
            if(hwloc_set_cpubind(
                   TaskFreeCtx::hwloc_ctx.topology,
                   obj->cpuset,
                   HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT))
            {
                char* str;
                int error = errno;
                hwloc_bitmap_asprintf(&str, obj->cpuset);
                SPDLOG_WARN("Couldn't cpubind to cpuset {}: {}\n", str, strerror(error));
                free(str);
            }
        }

        void membind()
        {
            if(hwloc_set_membind(
                   TaskFreeCtx::hwloc_ctx.topology,
                   obj->cpuset,
                   HWLOC_MEMBIND_BIND,
                   HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_STRICT))
            {
                char* str;
                int error = errno;
                hwloc_bitmap_asprintf(&str, obj->cpuset);
                SPDLOG_WARN("Couldn't membind to cpuset {}: {}\n", str, strerror(error));
                free(str);
            }
        }
    };

} // namespace redGrapes::dispatch::thread
