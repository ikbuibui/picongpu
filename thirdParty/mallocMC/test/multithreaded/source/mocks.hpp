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

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <functional>

inline auto constructAcc()
{
    static auto blockIdx = alpaka::Vec{size_t{0}};
    static auto blockCount = alpaka::Vec{size_t{1}};
    static alpaka::onAcc::cpu::detail::SharedStorage<1u> blockSharedMem{};
    static std::uint32_t dynSharedMemBytes = 0u;

    auto storage = alpaka::joinDict(
        alpaka::Dict{
            alpaka::DictEntry{
                alpaka::layer::block,
                alpaka::onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCount)}},
            alpaka::DictEntry{alpaka::layer::thread, alpaka::onAcc::cpu::OneLayer<alpaka::CVec<size_t, 1u>>{}},
            alpaka::DictEntry{alpaka::layer::shared, std::ref(blockSharedMem)},
            alpaka::DictEntry{alpaka::action::threadBlockSync, alpaka::onAcc::cpu::NoOp{}},
            alpaka::DictEntry{alpaka::object::launchedWidthFrameSpec, std::false_type{}},
            alpaka::DictEntry{alpaka::object::api, alpaka::api::host},
            alpaka::DictEntry{alpaka::object::deviceKind, alpaka::deviceKind::cpu},
            alpaka::DictEntry{alpaka::object::exec, alpaka::exec::cpuSerial},
            alpaka::DictEntry{alpaka::object::warpSize, std::integral_constant<std::uint32_t, 1u>{}}},
        alpaka::Dict{
            alpaka::DictEntry{alpaka::layer::dynShared, std::ref(blockSharedMem)},
            alpaka::DictEntry{alpaka::object::dynSharedMemBytes, std::ref(dynSharedMemBytes)}});

    using Acc = decltype(alpaka::onAcc::Acc(storage));
    return new Acc{storage};
}

//
static inline auto const accPointer = constructAcc();
static inline auto const& accSerial = *accPointer;

template<uint32_t T_blockSize, uint32_t T_pageSize, uint32_t T_wasteFactor = 1U, bool T_resetfreedpages = true>
struct HeapConfig
{
    static constexpr auto const accessblocksize = T_blockSize;
    static constexpr auto const pagesize = T_pageSize;
    static constexpr auto const wastefactor = T_wasteFactor;
    static constexpr auto const resetfreedpages = T_resetfreedpages;

    ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr static auto isInAllowedRange(
        auto const& /*acc*/,
        uint32_t const chunkSize,
        uint32_t const numBytes)
    {
        return (chunkSize >= numBytes && chunkSize <= T_wasteFactor * numBytes);
    }
};

struct AlignmentPolicy
{
    struct Properties
    {
        static constexpr uint32_t const dataAlignment = 1U;
    };
};
