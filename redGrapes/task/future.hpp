/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/*!
 * @file redGrapes/task_result.hpp
 */
#pragma once

#include "redGrapes/globalSpace.hpp"

namespace redGrapes
{


    /*!
     * Wrapper for std::future which consumes jobs
     * instead of waiting in get()
     */
    template<typename T, typename TTask>
    struct Future
    {
        Future(TTask& task) : taken(false), task(task)
        {
        }

        Future(Future&& other) : taken(other.taken), task(other.task)
        {
            SPDLOG_TRACE("MOVE future");
            other.taken = true;
        }

        ~Future()
        {
            if(!taken)
            {
                SPDLOG_TRACE("notify in destruct of future");
                task.get_result_get_event().notify();
            }
        }

        /*!
         * yields until the task has a valid result
         * and retrieves it.
         *
         * @return the result
         */
        T get(void)
        {
            // wait until result is set
            yield_impl<TTask>(task.get_result_set_event());

            // take result
            T result = std::move(*reinterpret_cast<T*>(task.get_result_data()));
            taken = true;
            task.get_result_get_event().notify();

            return result;
        }

        /*! check if the result is already computed
         */
        bool is_ready(void) const
        {
            return task.result_set_event.is_reached();
        }

    private:
        bool taken;
        TTask& task;
    }; // struct Future

    template<typename TTask>
    struct Future<void, TTask>
    {
        Future(TTask& task) : taken(false), task(task)
        {
        }

        Future(Future&& other) : taken(other.taken), task(other.task)
        {
            SPDLOG_TRACE("MOVE future");
            other.taken = true;
        }

        ~Future()
        {
            if(!taken)
            {
                SPDLOG_TRACE("notify in destruct of future");
                task.get_result_get_event().notify();
            }
        }

        /*!
         * yields until the task has a valid result
         * and retrieves it.
         *
         * @return the result
         */
        void get(void)
        {
            // wait until result is set
            yield_impl<TTask>(task.get_result_set_event());

            // take result
            taken = true;
            task.get_result_get_event().notify();
        }

        /*! check if the result is already computed
         */
        bool is_ready(void) const
        {
            return task.result_set_event.is_reached();
        }

    private:
        bool taken;
        TTask& task;
    };

} // namespace redGrapes
