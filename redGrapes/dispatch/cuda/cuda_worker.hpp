/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "redGrapes/dispatch/cuda/cuda_task_properties.hpp"
#include "redGrapes/dispatch/cuda/event_pool.hpp"
#include "redGrapes/globalSpace.hpp"
#include "redGrapes/scheduler/event.hpp"
#include "redGrapes/sync/cv.hpp"
#include "redGrapes/task/queue.hpp"

#include <cuda.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <queue>

namespace redGrapes::dispatch::cuda
{
    struct CudaStreamWrapper
    {
        cudaStream_t cuda_stream;

        CudaStreamWrapper()
        {
            cudaStreamCreate(&cuda_stream);
        }

        CudaStreamWrapper(CudaStreamWrapper const& other)
        {
            SPDLOG_WARN("CudaStreamWrapper copy constructor called!");
        }

        ~CudaStreamWrapper()
        {
            cudaStreamDestroy(cuda_stream);
        }
    };

    // this class is not thread safe
    // Stream dispatcher
    template<typename TTask>
    struct CudaWorker

    {
        using task_type = TTask;

        WorkerId id;
        std::vector<CudaStreamWrapper> streams;
        EventPool event_pool;

        /*! if true, the thread shall stop
         * instead of waiting when it is out of jobs
         */
        std::atomic_bool m_stop{false};
        std::atomic<unsigned> task_count{0};


        std::queue<std::pair<cudaEvent_t, scheduler::EventPtr<TTask>>> events;
        std::recursive_mutex mutex;

        //! condition variable for waiting if queue is empty
        CondVar cv;

        static constexpr size_t queue_capacity = 128;
        task::Queue<TTask> emplacement_queue{queue_capacity};
        task::Queue<TTask> ready_queue{queue_capacity};

        CudaWorker(WorkerId worker_id) : id(worker_id)
        {
        }

        CudaWorker(WorkerId worker_id, unsigned num_streams) : id{worker_id}, streams{num_streams}
        {
        }

        inline bool wake()
        {
            return cv.notify();
        }

        void stop()
        {
            SPDLOG_TRACE("Worker::stop()");
            m_stop.store(true, std::memory_order_release);
            wake();
        }

        /* adds a new task to the emplacement queue
         * and wakes up thread to kickstart execution
         */
        inline void dispatch_task(TTask& task)
        {
            emplacement_queue.push(&task);
            wake();
        }

        inline void execute_task(TTask& task)
        {
            TRACE_EVENT("Worker", "dispatch task");

            SPDLOG_DEBUG("cuda thread dispatch: execute task {}", task.task_id);
            assert(task.is_ready());
            std::lock_guard<std::recursive_mutex> lock(mutex);

            current_task = &task;

            // run the code that calls the CUDA API and submits work to *task->m_cuda_stream_idx
            auto event = task();

            cudaEvent_t cuda_event = event_pool.alloc();
            // works even if the m_cuda_stream index optional is nullopt, because it gets casted to 0
            cudaEventRecord(cuda_event, streams[*(task->m_cuda_stream_idx)].cuda_stream);
            auto my_event = create_event_impl<TTask>();
            events.push(std::make_pair(cuda_event, *my_event));
            SPDLOG_TRACE(
                "CudaStreamDispatcher {}: recorded event {}",
                streams[*(task->m_cuda_stream_idx)].cuda_stream,
                cuda_event);

            // TODO figure out the correct position for this
            task.get_pre_event().notify();

            if(event)
            {
                event->get_event().waker_id = id;
                task.sg_pause(*event);

                task.pre_event.up();
                task.get_pre_event().notify();
            }
            else
                task.get_post_event().notify();

            current_task = nullptr;
        }

        /* repeatedly try to find and execute tasks
         * until stop-flag is triggered by stop()
         */
        void work_loop()
        {
            SPDLOG_TRACE("Worker {} start work_loop()", this->id);
            while(!this->m_stop.load(std::memory_order_consume))
            {
                // this->cv.wait(); // TODO fix this by only waiting if event_pool is empty

                while(TTask* task = this->gather_task())
                {
                    execute_task(*task);
                    poll(); // TODO fix where to poll
                }
                poll();
            }
            SPDLOG_TRACE("Worker {} end work_loop()", this->id);
        }

        /* find a task that shall be executed next
         */
        TTask* gather_task()
        {
            {
                TRACE_EVENT("Worker", "gather_task()");
                TTask* task = nullptr;

                /* STAGE 1:
                 *
                 * first, execute all tasks in the ready queue
                 */
                SPDLOG_TRACE("Worker {}: consume ready queue", id);
                if((task = ready_queue.pop()))
                    return task;

                /* STAGE 2:
                 *
                 * after the ready queue is fully consumed,
                 * try initializing new tasks until one
                 * of them is found to be ready
                 */
                SPDLOG_TRACE("Worker {}: try init new tasks", id);
                while(this->init_dependencies(task, true))
                    if(task)
                        return task;

                return task;
            }
        }

        /*! take a task from the emplacement queue and initialize it,
         * @param t is set to the task if the new task is ready,
         * @param t is set to nullptr if the new task is blocked.
         * @param claimed if set, the new task will not be actiated,
         *        if it is false, activate_task will be called by notify_event
         *
         * @return false if queue is empty
         */
        bool init_dependencies(TTask*& t, bool claimed = true)
        {
            {
                TRACE_EVENT("Worker", "init_dependencies()");
                if(TTask* task = emplacement_queue.pop())
                {
                    SPDLOG_DEBUG("init task {}", task->task_id);

                    task->pre_event.up();
                    task->init_graph();

                    if(task->get_pre_event().notify(claimed))
                        t = task;
                    else
                    {
                        t = nullptr;
                    }

                    return true;
                }
                else
                    return false;
            }
        }

        //! checks if some cuda calls finished and notify the redGrapes manager
        void poll()
        {
            std::lock_guard<std::recursive_mutex> lock(mutex);
            if(!events.empty())
            {
                auto& cuda_event = events.front().first;
                auto& event = events.front().second;

                if(cudaEventQuery(cuda_event) == cudaSuccess)
                {
                    SPDLOG_TRACE("cuda event {} ready", cuda_event);
                    event_pool.free(cuda_event);
                    event.notify();

                    events.pop();
                }
            }
        }
    };

} // namespace redGrapes::dispatch::cuda

template<>
struct fmt::formatter<redGrapes::dispatch::cuda::CudaTaskProperties>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::dispatch::cuda::CudaTaskProperties const& prop, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "\"cuda_stream_idx\" : {}", *(prop.m_cuda_stream_idx));
    }
};
