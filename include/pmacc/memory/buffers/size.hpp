/*
 * This file is part of PMacc.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

#include <caravan/alpaka/operations.hpp>
#include <caravan/core/retained.hpp>

namespace pmacc
{
    /** Lazily copy a PMacc buffer's current size between host/device size storage. */
    template<typename T_Queue, typename T_Destination, typename T_Source>
    [[nodiscard]] auto size(T_Queue& queue, T_Destination destination, T_Source source)
    {
        using Source = std::remove_cvref_t<decltype(caravan::unwrap(source))>;
        return caravan::alpaka::copy(
            queue,
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<Source>, ::alpaka::Idx<Source>>::ones());
    }

    /** Queue-free lazy size copy; withDevice supplies the managed queue context. */
    template<typename T_Destination, typename T_Source>
    [[nodiscard]] auto size(T_Destination destination, T_Source source)
    {
        using Source = std::remove_cvref_t<decltype(caravan::unwrap(source))>;
        return caravan::alpaka::copy(
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<Source>, ::alpaka::Idx<Source>>::ones());
    }
} // namespace pmacc
