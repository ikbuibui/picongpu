/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/property/label.hpp
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"

#include <fmt/format.h>

#include <string>

namespace redGrapes
{

    struct LabelProperty
    {
        // TODO Optimization, use the workerId to allocate the label string seperate from the task
        // using string = std::basic_string<char, std::char_traits<char>, memory::StdAllocator<char>>;

        // Params workerId and scope_depth
        LabelProperty(WorkerId, unsigned)
        {
        }

        std::string label;

        template<typename TaskBuilder>
        struct Builder
        {
            TaskBuilder& builder;

            Builder(TaskBuilder& builder) : builder(builder)
            {
            }

            TaskBuilder& label(std::string const& l)
            {
                builder.task->label = l;
                return builder;
            }
        };

        struct Patch
        {
            template<typename PatchBuilder>
            struct Builder
            {
                Builder(PatchBuilder&)
                {
                }
            };
        };

        void apply_patch(Patch const&)
        {
        }
    };

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::LabelProperty>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::LabelProperty const& label_prop, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "\"label\" : \"{}\"", label_prop.label);
    }
};
