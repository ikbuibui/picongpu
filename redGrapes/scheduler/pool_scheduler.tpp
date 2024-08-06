/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/thread/DefaultWorker.hpp"
#include "redGrapes/scheduler/pool_scheduler.hpp"
#include "redGrapes/util/trace.hpp"

#include <spdlog/spdlog.h>

namespace redGrapes
{
    namespace scheduler
    {
        template<typename Worker>
        PoolScheduler<Worker>::PoolScheduler(WorkerId num_workers) : n_workers(num_workers)
                                                                   , m_worker_pool(num_workers)
        {
        }

        template<typename Worker>
        PoolScheduler<Worker>::PoolScheduler(dispatch::thread::WorkerPool<Worker> workerPool)
            : m_worker_pool(workerPool)
        {
        }

        /* send the new task to a worker
         */
        template<typename Worker>
        void PoolScheduler<Worker>::emplace_task(TTask& task)
        {
            // TODO: properly store affinity information in task
            WorkerId local_worker_id = task.worker_id - m_base_id;

            m_worker_pool.get_worker_thread(local_worker_id).worker.dispatch_task(task);

            /* hack as of 2023/11/17
             *
             * Additionally to the worker who got the new task above,
             * we will now notify another, available (idling) worker,
             * in trying to avoid stale tasks in cases where new tasks
             * are assigned to an already busy worker.
             */
#ifndef REDGRAPES_EMPLACE_NOTIFY_NEXT
#    define REDGRAPES_EMPLACE_NOTIFY_NEXT 0
#endif

#if REDGRAPES_EMPLACE_NOTIFY_NEXT
            auto id = m_worker_pool.probe_worker_by_state<WorkerId>(
                [&m_worker_pool](WorkerId idx)
                {
                    m_worker_pool.get_worker_thread(idx).worker.wake();
                    return idx;
                },
                dispatch::thread::WorkerState::AVAILABLE,
                local_worker_id,
                true);
#endif
        }

        /* send this already existing task to a worker,
         * but only through follower-list so it is not assigned to a worker yet.
         * since this task is now ready, send find a worker for it
         */
        template<typename Worker>
        void PoolScheduler<Worker>::activate_task(TTask& task)
        {
            //! worker id to use in case all workers are busy
            // TODO analyse and optimize
            static thread_local std::atomic<WorkerId> next_worker(
                TaskFreeCtx::current_worker_id ? *TaskFreeCtx::current_worker_id + 1 - m_base_id : 0);
            TRACE_EVENT("Scheduler", "activate_task");
            SPDLOG_TRACE("PoolScheduler::activate_task({})", task.task_id);

            int worker_id = m_worker_pool.find_free_worker();
            if(worker_id < 0)
            {
                worker_id = next_worker.fetch_add(1) % n_workers;
                if(worker_id == *TaskFreeCtx::current_worker_id)
                    worker_id = next_worker.fetch_add(1) % n_workers;
                m_worker_pool.get_worker_thread(worker_id).worker.ready_queue.push(&task);
            }
            else
            {
                m_worker_pool.get_worker_thread(worker_id).worker.ready_queue.push(&task);
                m_worker_pool.get_worker_thread(worker_id).worker.wake();
            }
        }

        /* Wakeup some worker or the main thread
         *
         * WakerId = WorkerId
         *
         * @return true if thread was indeed asleep
         */
        template<typename Worker>
        bool PoolScheduler<Worker>::wake(WakerId id)
        {
            auto local_waker_id = id - m_base_id;
            // TODO analyse and optimize
            if(local_waker_id > 0 && local_waker_id <= n_workers)
                return m_worker_pool.get_worker_thread(local_waker_id).worker.wake();
            else
                return false;
        }

        /* wakeup all workers
         */
        template<typename Worker>
        void PoolScheduler<Worker>::wake_all()
        {
            for(WorkerId i = m_base_id; i < m_base_id + n_workers; ++i)
                wake(i);
        }

        template<typename Worker>
        WorkerId PoolScheduler<Worker>::getNextWorkerID()
        {
            static std::atomic<WorkerId> local_next_worker_counter = 0;
            return (local_next_worker_counter++ % n_workers) + m_base_id;
        }

        template<typename Worker>
        void PoolScheduler<Worker>::init(WorkerId base_id)
        {
            // TODO check if it was already initalized
            m_base_id = base_id;
            m_worker_pool.emplace_workers(m_base_id);
        }

        template<typename Worker>
        void PoolScheduler<Worker>::startExecution()
        {
            // TODO check if it was already started
            m_worker_pool.start();
        }

        template<typename Worker>
        void PoolScheduler<Worker>::stopExecution()
        {
            // TODO check if it was already stopped
            m_worker_pool.stop();
        }


    } // namespace scheduler
} // namespace redGrapes
