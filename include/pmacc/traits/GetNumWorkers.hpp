/* Copyright 2017-2024 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/types.hpp"

#include <type_traits>
#include <utility>

namespace pmacc
{
    namespace traits
    {
        namespace detail
        {
            /** Device kind of the build's compute device (e.g. deviceKind::Cpu / NvidiaGpu / AmdGpu). */
            using ComputeDeviceKind = ALPAKA_TYPEOF(std::declval<ComputeDevice>().getDeviceKind());

            /** Whether an AnyExecutor launch on this build resolves to a seq executor.
             *
             * For every alpaka host (CPU) device, the executor alpaka::onHost::Queue selects for an
             * AnyExecutor launch is a seq executor (CpuSerial / CpuOmpBlocks / CpuTbbBlocks), all of
             * which launch with a block-thread extent of 1; GPU device kinds resolve to non-seq
             * executors. So the device kind is an exact compile-time proxy for "the resolved executor
             * is a seq executor" (alpaka::exec::isSeqExecutor_v). We key on the device kind rather than
             * the resolved executor type only because ComputeDevice::DeviceHandle (needed to call
             * alpaka::onHost::supportedExecutors at compile time) is private.
             */
            inline constexpr bool clampWorkersToOne
                = std::is_same_v<ComputeDeviceKind, ::alpaka::deviceKind::Cpu>
                  || std::is_same_v<ComputeDeviceKind, ::alpaka::deviceKind::NumaCpu>;
        } // namespace detail

        /** Get number of workers
         *
         * the number of workers for a kernel depending on the used accelerator
         *
         * @tparam T_maxWorkers the maximum number of workers
         * @return @p ::value number of workers
         *
         * @warning This is a stopgap and is intentionally brittle. It collapses to exactly
         *          1 worker for *any* seq executor and derives that executor from the global
         *          pmacc::ComputeDevice (i.e. it only models the AnyExecutor launch path).
         *          This breaks as soon as either of the following becomes true:
         *            - a CPU/host alpaka executor is selected that runs more than one thread
         *              per block: the real worker count is then that thread count, not 1, and
         *              hard-coding 1 leaves the other threads idle (under-subscription); or
         *            - the lockstep launch gains the planned per-kernel-launch executor
         *              option (a compile-time T_Executor): this trait ignores that per-launch
         *              executor and keeps clamping based on ComputeDevice, so a launch with an
         *              executor whose real worker count differs from both 1 and blockDomSize
         *              will be wrong.
         *          The correct fix is to key the worker count on the *actual* executor used
         *          for the launch and on that executor's real threads-per-block, instead of
         *          this global seq/non-seq catch-all. See @ref detail::clampWorkersToOne.
         */
        template<uint32_t T_maxWorkers>
        struct GetNumWorkers
        {
            static constexpr uint32_t value = detail::clampWorkersToOne ? 1u : T_maxWorkers;
        };
    } // namespace traits
} // namespace pmacc
