/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <tuple>
#include <utility>

#include <caravan/alpaka.hpp>

namespace pmacc::async
{
    /** Alpaka view plus the allocation handle retained by an operation. */
    template<typename T_View, typename T_Allocation>
    struct OwnedView
    {
        T_View view;
        T_Allocation allocation;
    };

    /** Kernel argument plus an allocation handle retained only for lifetime. */
    template<typename T_Argument, typename T_Allocation>
    struct Retained
    {
        T_Argument argument;
        T_Allocation allocation;
    };

    template<typename T_Argument, typename T_View, typename T_Allocation>
    auto retain(T_Argument argument, OwnedView<T_View, T_Allocation> const& owner)
    {
        return Retained<T_Argument, T_Allocation>{std::move(argument), owner.allocation};
    }

    template<typename T_Queue, typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Queue& queue, T_Destination destination, T_Source source, T_Extent extent)
    {
        return caravan::alpaka::submit(
            queue,
            [destination = std::move(destination), source = std::move(source), extent](T_Queue& nativeQueue) mutable
            { ::alpaka::memcpy(nativeQueue, destination.view, source.view, extent); });
    }

    template<typename T_Queue, typename T_View, typename T_Allocation>
    auto fill(T_Queue& queue, OwnedView<T_View, T_Allocation> destination, std::uint8_t byte)
    {
        return caravan::alpaka::submit(
            queue,
            [destination = std::move(destination), byte](T_Queue& nativeQueue) mutable
            { ::alpaka::memset(nativeQueue, destination.view, byte); });
    }

    template<typename T_Queue, typename T_Destination, typename T_Source>
    auto size(T_Queue& queue, T_Destination destination, T_Source source)
    {
        using SourceView = decltype(source.view);
        return copy(
            queue,
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<SourceView>, ::alpaka::Idx<SourceView>>::ones());
    }

    namespace detail
    {
        template<typename T>
        decltype(auto) nativeArgument(T& value)
        {
            return (value);
        }

        template<typename T_Argument, typename T_Allocation>
        T_Argument& nativeArgument(Retained<T_Argument, T_Allocation>& value)
        {
            return value.argument;
        }
    } // namespace detail

    /** Lazy kernel launch retaining any Retained allocation arguments. */
    template<typename T_Acc, typename T_Queue, typename T_WorkDiv, typename T_Kernel, typename... T_Args>
    auto kernel(T_Queue& queue, T_WorkDiv workDiv, T_Kernel kernel, T_Args... args)
    {
        return caravan::alpaka::submit(
            queue,
            [workDiv = std::move(workDiv),
             kernel = std::move(kernel),
             args = std::tuple<T_Args...>{std::move(args)...}](T_Queue& nativeQueue) mutable
            {
                std::apply(
                    [&](auto&... values)
                    { ::alpaka::exec<T_Acc>(nativeQueue, workDiv, kernel, detail::nativeArgument(values)...); },
                    args);
            });
    }
} // namespace pmacc::async
