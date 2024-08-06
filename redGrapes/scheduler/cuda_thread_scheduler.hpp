/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/dispatch/cuda/cuda_worker.hpp"
#include "redGrapes/globalSpace.hpp"
#include "redGrapes/scheduler/thread_scheduler.hpp"

#include <atomic>
#include <cassert>

namespace redGrapes
{
    namespace scheduler
    {
        template<typename TTask>
        class CudaThreadScheduler : public ThreadScheduler<dispatch::cuda::CudaWorker<TTask>>
        {
        private:
            unsigned num_streams;

        public:
            CudaThreadScheduler(unsigned num_streams = 1) : num_streams{num_streams}
            {
            }

            void init(WorkerId base_id) override
            {
                this->m_base_id = base_id;
                // TODO check if it was already initalized
                if(!this->m_worker_thread)
                {
                    WorkerId pu_id = base_id % TaskFreeCtx::n_pus;
                    // allocate worker with id `i` on arena `i`,
                    hwloc_obj_t obj = hwloc_get_obj_by_type(TaskFreeCtx::hwloc_ctx.topology, HWLOC_OBJ_PU, pu_id);
                    TaskFreeCtx::worker_alloc_pool.allocs.emplace_back(
                        memory::HwlocAlloc(TaskFreeCtx::hwloc_ctx, obj),
                        REDGRAPES_ALLOC_CHUNKSIZE);

                    this->m_worker_thread
                        = memory::alloc_shared_bind<dispatch::thread::WorkerThread<dispatch::cuda::CudaWorker<TTask>>>(
                            this->m_base_id,
                            obj,
                            this->m_base_id,
                            num_streams);
                }
            }

            /*! whats the task dependency type for the edge a -> b (task a precedes task b)
             * @return true if task b depends on the pre event of task a, false if task b depends on the post event
             * of task b.
             */
            bool task_dependency_type(TTask const& a, TTask const& b)
            {
                if(a.m_cuda_stream_idx)
                    return true;
                else
                    return false;
                ;
            }

            /**
             * Only to be used if the user wants to manage streams directly
             * The user must ensure that if this method is used, they must set the cuda_stream_index() property
             */
            cudaStream_t getCudaStreamIdx(unsigned idx) const
            {
                assert(idx < num_streams);
                return this->m_worker_thread->worker.streams[idx].cuda_stream;
            }

            /**
             * Returns the cuda stream to the user to use in their cuda kernel
             * Also sets the stream index on the task which calls this method
             * requires current_task is not nullptr
             */
            cudaStream_t getCudaStream() const
            {
                static std::atomic_uint stream_idx = 0;
                auto task_stream_idx = stream_idx.fetch_add(1) % num_streams;
                static_cast<TTask*>(current_task)->m_cuda_stream_idx = task_stream_idx;
                return this->m_worker_thread->worker.streams[task_stream_idx].cuda_stream;
            }
        };
    } // namespace scheduler
} // namespace redGrapes
