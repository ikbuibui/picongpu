/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Benjamin Worpitz
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


#include "pmacc/async/Operations.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/exec/KernelLauncher.hpp"
#include "pmacc/traits/GetNComponents.hpp"
#include "pmacc/types.hpp"

namespace pmacc::exec::detail
{
    template<typename T_Kernel, uint32_t T_dim>
    struct KernelLauncher
    {
        //! kernel functor
        T_Kernel const m_kernel;
        //! grid extents for the kernel
        math::Vector<IdxType, T_dim> const m_gridExtent;
        //! block extents for the kernel
        math::Vector<IdxType, T_dim> const m_blockExtent;

        /** kernel starter object
         *
         * @param kernel pmacc Kernel
         */
        template<typename T_VectorGrid, typename T_VectorBlock>
        HINLINE KernelLauncher(
            T_Kernel const& kernel,
            T_VectorGrid const& gridExtent,
            T_VectorBlock const& blockExtent)
            : m_kernel(kernel)
            , m_gridExtent(gridExtent)
            , m_blockExtent(blockExtent)
        {
        }

        /** Enqueue this kernel from a submission state that retains all arguments through completion. */
        template<typename T_Queue, typename... T_Args>
        HINLINE void enqueueNative(T_Queue& queue, T_Args&&... args) const
        {
            auto const gridExtent = m_gridExtent.toAlpakaKernelVec();
            auto const blockExtent = m_blockExtent.toAlpakaKernelVec();
            auto const elemExtent = math::Vector<IdxType, T_dim>::create(1).toAlpakaKernelVec();
            auto const workDiv
                = ::alpaka::WorkDivMembers<::alpaka::DimInt<T_dim>, IdxType>(gridExtent, blockExtent, elemExtent);
            ::alpaka::exec<Acc<T_dim>>(queue, workDiv, m_kernel, pmacc::async::detail::nativeArgument(args)...);
        }

        /** Lazily describe this kernel on an explicitly borrowed queue. */
        template<typename T_Queue, typename... T_Args>
        HINLINE auto sender(T_Queue& queue, T_Args... args) const
        {
            return caravan::alpaka::submit(
                queue,
                [launcher = *this, args = std::tuple<T_Args...>{std::move(args)...}](T_Queue& nativeQueue) mutable
                { std::apply([&](auto&... values) { launcher.enqueueNative(nativeQueue, values...); }, args); });
        }
    };

} // namespace pmacc::exec::detail
