/* Copyright 2026 Rene Widera
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

#include "pmacc/communication/CommunicatorMPI.hpp"
#include "pmacc/eventSystem/Manager.hpp"

#include <functional>
#include <utility>

namespace pmacc::eventSystem
{
    /** Wait at a barrier while keeping PMacc-side completions moving. */
    template<unsigned DIM, typename T_Progress = std::function<void()>>
    void mpiBlocking(CommunicatorMPI<DIM>& communicator, T_Progress progress = [] {})
    {
        async::Context context;
        auto barrier = context.spawn(communicator.barrier());
        Manager::getInstance().waitFor(
            [&]()
            {
                std::invoke(progress);
                context.runReady();
                if(barrier.state() == caravan::CompletionState::pending)
                    return false;
                barrier.wait();
                return true;
            });
    }
} // namespace pmacc::eventSystem
