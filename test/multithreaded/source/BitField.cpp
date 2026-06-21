/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2024 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Julian Johannes Lenz

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

#include "mocks.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <mallocMC/creationPolicies/FlatterScatter/BitField.hpp>

#include <cstdint>
#include <stop_token>
#include <thread>

using mallocMC::CreationPolicies::FlatterScatterAlloc::BitMask;
using mallocMC::CreationPolicies::FlatterScatterAlloc::BitMaskSize;
using namespace std::chrono_literals;

TEST_CASE("Threaded BitMask")
{
    BitMask mask{};

    SECTION("finds first free bit despite noise.")
    {
        // This is a regression test. An earlier version of this algorithm used to fail when other parts of the bit
        // mask experienced frequent change during the search. We simulate this by letting a "noise thread" toggle
        // unimportant bits while a "search thread" tries to find the first free bit. While the noise does not affect
        // the result, a previous version of the algorithm does fail under these conditions (as verified by
        // experiment).

        uint32_t const firstFreeIndex = GENERATE(0U, 1U, 10U);
        for(uint32_t i = 0; i < firstFreeIndex; ++i)
        {
            mask.set(accSerial, i);
        }

        uint32_t result = BitMaskSize;
        auto noiseThread = std::jthread(
            [&mask, firstFreeIndex](std::stop_token const& stopToken)
            {
                while(not stopToken.stop_requested())
                {
                    for(uint32_t i = firstFreeIndex + 1; i < BitMaskSize; ++i)
                    {
                        mask.flip(accSerial, i);
                    }
                }
            });
        std::thread([&mask, &result]() { result = mask.firstFreeBit(accSerial); }).join();
        std::this_thread::sleep_for(20ms);
        CHECK(result == firstFreeIndex);
        noiseThread.request_stop();
    }
}
