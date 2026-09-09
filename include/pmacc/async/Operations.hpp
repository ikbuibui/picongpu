/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <caravan/alpaka.hpp>

namespace pmacc::async
{
    using caravan::alpaka::copy;
    using caravan::alpaka::fill;
    using caravan::alpaka::kernel;
    using caravan::alpaka::OwnedView;
    using caravan::alpaka::retain;
    using caravan::alpaka::Retained;
    using caravan::alpaka::size;

    namespace detail
    {
        using caravan::alpaka::detail::nativeArgument;
    } // namespace detail
} // namespace pmacc::async
