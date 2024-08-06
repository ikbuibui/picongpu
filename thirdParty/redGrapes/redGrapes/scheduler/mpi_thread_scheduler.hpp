/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/dispatch/mpi/mpiWorker.hpp"
#include "redGrapes/dispatch/mpi/request_pool.hpp"
#include "redGrapes/scheduler/thread_scheduler.hpp"

#include <memory>

namespace redGrapes
{
    namespace scheduler
    {

        template<typename TTask>
        struct MPIThreadScheduler : public ThreadScheduler<dispatch::mpi::MPIWorker<TTask>>
        {
            // if worker is  MPI worker
            std::shared_ptr<dispatch::mpi::RequestPool<TTask>> getRequestPool()
            {
                return this->m_worker_thread->worker.requestPool;
            }
        };


    } // namespace scheduler

} // namespace redGrapes
