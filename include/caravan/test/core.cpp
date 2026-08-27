/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <array>
#include <atomic>
#include <cassert>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include <caravan/core.hpp>

namespace
{
    struct InlineExecutor
    {
        template<typename T_Function>
        void post(T_Function&& function)
        {
            std::forward<T_Function>(function)();
        }
    };

    struct RecursionTrackingExecutor
    {
        template<typename T_Function>
        void post(T_Function&& function)
        {
            ++depth;
            if(depth > maxDepth)
                maxDepth = depth;
            std::forward<T_Function>(function)();
            --depth;
        }

        unsigned depth = 0u;
        unsigned maxDepth = 0u;
    };

    void testCompletionAndContinuations()
    {
        InlineExecutor executor;
        caravan::EventSource source;
        unsigned calls = 0u;
        auto first = source.event().then(executor, [&] { ++calls; });
        assert(source.setReady());
        assert(!source.setReady());
        first.wait();

        auto second = source.event().then(executor, [&] { ++calls; });
        second.wait();
        assert(calls == 2u);
    }

    void testNoRecursiveInlineChains()
    {
        RecursionTrackingExecutor executor;
        caravan::EventSource source;
        auto tail = source.event();
        for(unsigned i = 0u; i < 1000u; ++i)
            tail = tail.then(executor, [] {});
        source.setReady();
        tail.wait();
        assert(executor.maxDepth == 1u);
    }

    void testWhenAllAndFailure()
    {
        caravan::EventSource first;
        caravan::EventSource second;
        std::array events{first.event(), second.event()};
        auto joined = caravan::whenAll(events);
        first.setReady();
        assert(joined.state() == caravan::CompletionState::pending);
        second.setReady();
        joined.wait();

        caravan::EventSource failedInput;
        caravan::EventSource unfinishedInput;
        std::array failingEvents{failedInput.event(), unfinishedInput.event()};
        auto failedJoin = caravan::whenAll(failingEvents);
        failedInput.setFailed(std::make_exception_ptr(std::runtime_error("joined failure")));
        assert(failedJoin.state() == caravan::CompletionState::pending);
        unfinishedInput.setReady();
        try
        {
            failedJoin.wait();
            assert(false);
        }
        catch(std::runtime_error const& error)
        {
            assert(std::string_view{error.what()} == "joined failure");
        }

        InlineExecutor executor;
        caravan::EventSource failed;
        bool called = false;
        auto successor = failed.event().then(executor, [&] { called = true; });
        failed.setFailed(std::make_exception_ptr(std::runtime_error("expected")));
        try
        {
            successor.wait();
            assert(false);
        }
        catch(std::runtime_error const& error)
        {
            assert(std::string_view{error.what()} == "expected");
        }
        assert(!called);
    }

    void testFuture()
    {
        InlineExecutor executor;
        caravan::Promise<int> promise;
        auto doubled = promise.future().then(executor, [](int value) { return value * 2; });
        promise.setValue(21);
        assert(doubled.result() == 42);
    }

    void testExactlyOnceCompletion()
    {
        caravan::EventSource source;
        std::atomic<unsigned> winners = 0u;
        std::vector<std::thread> threads;
        for(unsigned i = 0u; i < 8u; ++i)
            threads.emplace_back([&] { winners.fetch_add(source.setReady()); });
        for(auto& thread : threads)
            thread.join();
        assert(winners == 1u);
    }

    void testRegistrationRace()
    {
        InlineExecutor executor;
        caravan::EventSource source;
        constexpr unsigned threadCount = 8u;
        constexpr unsigned continuationsPerThread = 100u;
        std::atomic<unsigned> calls = 0u;
        std::vector<caravan::Event> completions;
        std::mutex completionsMutex;
        std::vector<std::thread> threads;

        for(unsigned thread = 0u; thread < threadCount; ++thread)
        {
            threads.emplace_back(
                [&]
                {
                    std::vector<caravan::Event> local;
                    for(unsigned i = 0u; i < continuationsPerThread; ++i)
                        local.emplace_back(source.event().then(executor, [&] { calls.fetch_add(1u); }));
                    std::lock_guard lock(completionsMutex);
                    completions.insert(completions.end(), local.begin(), local.end());
                });
        }
        std::thread completer([&] { source.setReady(); });
        for(auto& thread : threads)
            thread.join();
        completer.join();
        caravan::whenAll(completions).wait();
        assert(calls == threadCount * continuationsPerThread);
    }

    void testExecutorWaitGuard()
    {
        InlineExecutor executor;
        caravan::EventSource start;
        caravan::EventSource pending;
        auto blocked = start.event().then(executor, [&] { pending.event().wait(); });
        start.setReady();
        try
        {
            blocked.wait();
            assert(false);
        }
        catch(std::logic_error const&)
        {
        }
    }
} // namespace

int main()
{
    assert(caravan::readyEvent().isReady());
    testCompletionAndContinuations();
    testNoRecursiveInlineChains();
    testWhenAllAndFailure();
    testFuture();
    testExactlyOnceCompletion();
    testRegistrationRace();
    testExecutorWaitGuard();
}
