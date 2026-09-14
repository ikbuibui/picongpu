/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <array>
#include <atomic>
#include <cassert>
#include <future>
#include <memory>
#include <stdexcept>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

int main()
{
    using Queue = alpaka::QueueCpuNonBlocking;
    auto const device = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
    try
    {
        caravan::alpaka::SharedQueuePool<Queue> invalid{device, 0u};
        assert(false);
    }
    catch(std::invalid_argument const&)
    {
    }

    // Nested logical forks keep their dependencies when fixed physical lanes alias.
    for(std::size_t count : {1u, 2u, 3u})
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, count};
        std::array<int, 4u> values{};
        int seed = 0;
        auto branch = [&](std::size_t index)
        {
            return caravan::alpaka::enqueue(
                [&, index]
                {
                    assert(seed == 42);
                    if(index >= 2u)
                        assert(values[1u] == seed);
                    values[index] = seed;
                });
        };
        auto graph = caravan::alpaka::enqueue([&] { seed = 42; })
                     | caravan::alpaka::sequence(
                         caravan::whenAll(
                             branch(0u),
                             branch(1u) | caravan::alpaka::sequence(caravan::whenAll(branch(2u), branch(3u)))))
                     | caravan::alpaka::sequence(
                         caravan::alpaka::enqueue([&] { assert((values == std::array{42, 42, 42, 42})); }));
        caravan::syncWait(caravan::alpaka::withDevice(pool, std::move(graph)));
    }

    // The fixed cap allows two queues to run while further work aliases a busy queue.
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 2u};
        caravan::AsyncScope scope;
        std::promise<void> release, firstTwoStarted;
        auto gate = release.get_future().share();
        auto started = firstTwoStarted.get_future();
        std::atomic<unsigned> entered = 0u;
        auto blocked = [&]
        {
            return caravan::alpaka::withDevice(
                pool,
                caravan::alpaka::enqueue(
                    [&, gate]
                    {
                        if(++entered == 2u)
                            firstTwoStarted.set_value();
                        gate.wait();
                    }));
        };
        auto first = scope.spawn(blocked());
        auto second = scope.spawn(blocked());
        auto third = scope.spawn(blocked());
        started.get();
        assert(entered == 2u);
        assert(first.state() == caravan::CompletionState::pending);
        assert(second.state() == caravan::CompletionState::pending);
        assert(third.state() == caravan::CompletionState::pending);
        release.set_value();
        first.wait();
        second.wait();
        third.wait();
        assert(entered == 3u);
        scope.join().wait();
    }

    // Errors and retained storage stay isolated until earlier work on a shared queue is quiescent.
    {
        caravan::alpaka::SharedQueuePool<Queue> pool{device, 1u};
        caravan::AsyncScope scope;
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        auto started = entered.get_future();
        auto storage = std::make_shared<int>(42);
        std::weak_ptr<int> retained = storage;
        auto bad = scope.spawn(
            caravan::alpaka::withDevice(
                pool,
                caravan::alpaka::enqueue(
                    [storage = std::move(storage), gate, &entered]
                    {
                        entered.set_value();
                        gate.wait();
                        assert(*storage == 42);
                        throw std::runtime_error("bad graph");
                    })));
        started.get();
        bool goodRan = false;
        auto good = scope.spawn(caravan::alpaka::withDevice(pool, caravan::alpaka::enqueue([&] { goodRan = true; })));
        assert(!retained.expired() && !goodRan);
        assert(bad.state() == caravan::CompletionState::pending);
        assert(good.state() == caravan::CompletionState::pending);
        release.set_value();
        try
        {
            bad.wait();
            assert(false);
        }
        catch(std::runtime_error const&)
        {
        }
        good.wait();
        scope.join().wait();
        assert(goodRan && retained.expired());
    }

    // Independently bound pool strategies compose without exposing their queues.
    {
        caravan::alpaka::SharedQueuePool<Queue> first{device, 1u}, second{device, 1u};
        caravan::alpaka::QueuePool<Queue> leased{device};
        std::atomic<unsigned> ran = 0u;
        caravan::syncWait(
            caravan::whenAll(
                caravan::alpaka::withDevice(first, caravan::alpaka::enqueue([&] { ++ran; })),
                caravan::alpaka::withDevice(second, caravan::alpaka::enqueue([&] { ++ran; })),
                caravan::alpaka::withDevice(leased, caravan::alpaka::enqueue([&] { ++ran; }))));
        assert(ran == 3u);
    }
}
