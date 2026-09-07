/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <cstddef>

#include <caravan/mpi/context.hpp>

namespace caravan
{
    struct SendResult
    {
        std::size_t bytes;
    };

    struct ReceiveResult
    {
        Peer source;
        MessageTag tag;
        std::size_t bytes;
    };

    struct AllReduceResult
    {
        std::size_t elements;
    };

    struct ReduceResult
    {
        std::size_t elements;
    };

    struct GatherResult
    {
        std::size_t bytes;
    };
} // namespace caravan
