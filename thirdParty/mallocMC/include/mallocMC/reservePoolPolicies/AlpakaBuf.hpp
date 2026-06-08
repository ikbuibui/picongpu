/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2020-2024 Helmholtz-Zentrum Dresden - Rossendorf,
                 CERN

  Author(s):  Bernhard Manfred Gruber
              Julian J. Lenz - j.lenz ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

#include <alpaka/alpaka.hpp>

#include <any>
#include <string>

namespace mallocMC
{
    namespace ReservePoolPolicies
    {
        struct AlpakaBuf
        {
            template<typename AlpakaDev>
            auto setMemPool(AlpakaDev const& dev, size_t memsize) -> void*
            {
                auto buffer = alpaka::onHost::alloc<unsigned char>(dev, memsize);
                pool = alpaka::onHost::data(buffer);
                poolStorage.emplace<decltype(buffer)>(std::move(buffer));
                return pool;
            }

            void resetMemPool()
            {
                poolStorage.reset();
                pool = nullptr;
            }

            static auto classname() -> std::string
            {
                return "AlpakaBuf";
            }

        private:
            std::any poolStorage;
            unsigned char* pool = nullptr;
        };
    } // namespace ReservePoolPolicies
} // namespace mallocMC
