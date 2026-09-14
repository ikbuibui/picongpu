/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <caravan/alpaka/queue/managed.hpp>
#include <caravan/alpaka/submission.hpp>
#include <caravan/core/retained.hpp>

namespace caravan::alpaka
{
    /** Queue-free lazy enqueue of a native alpaka task. */
    template<typename T_Task>
    auto enqueue(T_Task task)
    {
        return detail::managedSubmit([task = std::move(task)](auto& nativeQueue) mutable
                                     { ::alpaka::enqueue(nativeQueue, task); });
    }

    /** Lazy byte fill. The buffer/view and any explicit owner are retained by value. */
    template<typename T_Queue, typename T_Buffer>
    auto fill(T_Queue& queue, T_Buffer buffer, std::uint8_t byte)
    {
        return submit(
            queue,
            [buffer = std::move(buffer), byte](T_Queue& nativeQueue) mutable
            { ::alpaka::memset(nativeQueue, caravan::unwrap(buffer), byte); });
    }

    /** Queue-free lazy byte fill; withDevice supplies the managed queue context. */
    template<typename T_Buffer>
    auto fill(T_Buffer buffer, std::uint8_t byte)
    {
        return detail::managedSubmit([buffer = std::move(buffer), byte](auto& nativeQueue) mutable
                                     { ::alpaka::memset(nativeQueue, caravan::unwrap(buffer), byte); });
    }

    /** Lazy copy. Buffer/views, explicit owners, and the extent are retained by value. */
    template<typename T_Queue, typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Queue& queue, T_Destination destination, T_Source source, T_Extent extent)
    {
        return submit(
            queue,
            [destination = std::move(destination), source = std::move(source), extent](T_Queue& nativeQueue) mutable
            { ::alpaka::memcpy(nativeQueue, caravan::unwrap(destination), caravan::unwrap(source), extent); });
    }

    /** Queue-free lazy copy; withDevice supplies the managed queue context. */
    template<typename T_Destination, typename T_Source, typename T_Extent>
    auto copy(T_Destination destination, T_Source source, T_Extent extent)
    {
        return detail::managedSubmit(
            [destination = std::move(destination), source = std::move(source), extent](auto& nativeQueue) mutable
            { ::alpaka::memcpy(nativeQueue, caravan::unwrap(destination), caravan::unwrap(source), extent); });
    }

    /** Lazy one-element copy for size values. */
    template<typename T_Queue, typename T_Destination, typename T_Source>
    auto size(T_Queue& queue, T_Destination destination, T_Source source)
    {
        using Source = std::remove_cvref_t<decltype(caravan::unwrap(source))>;
        return copy(
            queue,
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<Source>, ::alpaka::Idx<Source>>::ones());
    }

    /** Queue-free lazy one-element copy for size values. */
    template<typename T_Destination, typename T_Source>
    auto size(T_Destination destination, T_Source source)
    {
        using Source = std::remove_cvref_t<decltype(caravan::unwrap(source))>;
        return caravan::alpaka::copy(
            std::move(destination),
            std::move(source),
            ::alpaka::Vec<::alpaka::Dim<Source>, ::alpaka::Idx<Source>>::ones());
    }

    /** Lazy kernel launch retaining work division, kernel, arguments, and explicit owners. */
    template<typename T_Acc, typename T_Queue, typename T_WorkDiv, typename T_Kernel, typename... T_Args>
    requires(::alpaka::isQueue<T_Queue>)
    auto kernel(T_Queue& queue, T_WorkDiv workDiv, T_Kernel kernel, T_Args... args)
    {
        return submit(
            queue,
            [workDiv = std::move(workDiv),
             kernel = std::move(kernel),
             args = std::tuple<T_Args...>{std::move(args)...}](T_Queue& nativeQueue) mutable
            {
                std::apply(
                    [&](auto&... values)
                    { ::alpaka::exec<T_Acc>(nativeQueue, workDiv, kernel, caravan::unwrap(values)...); },
                    args);
            });
    }

    /** Queue-free lazy kernel launch; withDevice supplies the managed queue context. */
    template<typename T_Acc, typename T_WorkDiv, typename T_Kernel, typename... T_Args>
    auto kernel(T_WorkDiv workDiv, T_Kernel kernel, T_Args... args)
    {
        return detail::managedSubmit(
            [workDiv = std::move(workDiv),
             kernel = std::move(kernel),
             args = std::tuple<T_Args...>{std::move(args)...}](auto& nativeQueue) mutable
            {
                std::apply(
                    [&](auto&... values)
                    { ::alpaka::exec<T_Acc>(nativeQueue, workDiv, kernel, caravan::unwrap(values)...); },
                    args);
            });
    }
} // namespace caravan::alpaka
