/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <mpi.h>

namespace caravan::detail
{
    inline std::runtime_error mpiError(char const* operation, int errorCode)
    {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(errorCode, message, &length);
        return std::runtime_error(
            std::string{operation} + ": " + std::string{message, static_cast<std::size_t>(length)});
    }
} // namespace caravan::detail
