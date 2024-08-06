/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <optional>

namespace redGrapes
{
    namespace dispatch
    {
        namespace cuda
        {

            struct CudaTaskProperties
            {
                std::optional<unsigned> m_cuda_stream_idx;

                CudaTaskProperties()
                {
                }

                template<typename PropertiesBuilder>
                struct Builder
                {
                    PropertiesBuilder& builder;

                    Builder(PropertiesBuilder& b) : builder(b)
                    {
                    }

                    PropertiesBuilder& cuda_stream_index(unsigned cuda_stream_idx)
                    {
                        *(builder.task->m_cuda_stream_idx) = cuda_stream_idx;
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

                void apply_patch(Patch const&){};
            };

        } // namespace cuda
    } // namespace dispatch
} // namespace redGrapes
