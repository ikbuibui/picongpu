/* Copyright 2022-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/task/property/graph.hpp"
#include "redGrapes/task/property/id.hpp"
#include "redGrapes/task/property/inherit.hpp"
#include "redGrapes/task/property/resource.hpp"
#include "redGrapes/task/task_base.hpp"

#include <cstdint>
#include <type_traits>

namespace redGrapes
{

    template<typename T>
    concept C_TaskProperty = requires(T taskProp, typename T::Patch patch) {
        { taskProp.apply_patch(patch) } -> std::same_as<void>;
    };

    template<C_TaskProperty... UserTaskProperties>
    struct Task
        : TaskBase<Task<UserTaskProperties...>>
        , TaskProperties1<
              GraphProperty<Task<UserTaskProperties...>>,
              ResourceProperty<Task<UserTaskProperties...>>,
              IDProperty,
              UserTaskProperties...>
    {
        using TaskProperties = TaskProperties1<
            GraphProperty<Task<UserTaskProperties...>>,
            ResourceProperty<Task<UserTaskProperties...>>,
            IDProperty,
            UserTaskProperties...>;

        virtual ~Task()
        {
        }

        // worker id where task is first emplaced and task memory is located (may be stolen later)
        WorkerId worker_id;
        std::atomic<uint8_t> removal_countdown;
        scheduler::IScheduler<Task<UserTaskProperties...>>* scheduler_p;

        Task(WorkerId _worker_id, unsigned scope_depth, scheduler::IScheduler<Task<UserTaskProperties...>>& scheduler)
            : TaskProperties(_worker_id, scope_depth)
            , worker_id(_worker_id)
            , removal_countdown(2)
            , scheduler_p(&scheduler)
        {
        }

        virtual void* get_result_data()
        {
            return nullptr;
        }
    };

    // TODO: fuse ResultTask and FunTask into one template
    //     ---> removes one layer of virtual function calls

    template<typename Result, typename TTask>
    struct ResultTask : TTask
    {
        Result result_data;

        ResultTask(WorkerId worker_id, unsigned scope_depth, scheduler::IScheduler<TTask>& scheduler)
            : TTask(worker_id, scope_depth, scheduler)
        {
        }

        virtual ~ResultTask()
        {
        }

        virtual void* get_result_data()
        {
            return &result_data;
        }

        virtual Result run_result() = 0;

        void run() final
        {
            result_data = run_result();
            this->get_result_set_event().notify(); // result event now ready
        }
    };

    template<typename TTask>
    struct ResultTask<void, TTask> : TTask
    {
        ResultTask(WorkerId worker_id, unsigned scope_depth, scheduler::IScheduler<TTask>& scheduler)
            : TTask(worker_id, scope_depth, scheduler)
        {
        }

        virtual ~ResultTask()
        {
        }

        virtual void run_result()
        {
        }

        void run() final
        {
            run_result();
            this->get_result_set_event().notify();
        }
    };

    template<typename F, typename TTask>
    struct FunTask : ResultTask<typename std::invoke_result_t<F>, TTask>
    {
        FunTask(WorkerId worker_id, unsigned scope_depth, scheduler::IScheduler<TTask>& scheduler)
            : ResultTask<typename std::invoke_result_t<F>, TTask>(worker_id, scope_depth, scheduler)
        {
        }

        std::optional<F> impl;

        virtual ~FunTask()
        {
        }

        typename std::invoke_result_t<F> run_result()
        {
            return (*this->impl)();
        }
    };

} // namespace redGrapes
