/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/async/Context.hpp>

#include <chrono>
#include <thread>

#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PMacc async context owns work and drives host continuations", "[async]")
{
    pmacc::async::Context context;
    auto const applicationThread = std::this_thread::get_id();
    std::thread::id completionThread;
    bool ran = false;
    caravan::EventSource backendCompletion;

    auto operation = context.spawn(
        caravan::then(
            context.onControl(caravan::asSender(backendCompletion.event())),
            [&]
            {
                ran = true;
                CHECK(std::this_thread::get_id() == applicationThread);
                CHECK(std::this_thread::get_id() != completionThread);
            }));

    std::thread backend(
        [&]
        {
            completionThread = std::this_thread::get_id();
            backendCompletion.setReady();
        });
    CHECK(operation.state() == caravan::CompletionState::pending);
    context.wait(operation);
    backend.join();
    CHECK(ran);

    caravan::EventSource externalCompletion;
    std::thread external(
        [&]
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            externalCompletion.setReady();
        });
    context.wait(externalCompletion.event());
    external.join();

    caravan::EventSource progressed;
    size_t progressCalls = 0u;
    context.wait(
        progressed.event(),
        [&]
        {
            ++progressCalls;
            progressed.setReady();
        });
    CHECK(progressCalls == 1u);

    caravan::EventSource pending;
    auto checked = context.spawn(
        caravan::then(
            context.scheduler().schedule(),
            [&] { CHECK_THROWS_AS(context.wait(pending.event()), std::logic_error); }));
    context.wait(checked);
}

TEST_CASE("PMacc wait wakes for every terminal channel and survives progress errors", "[async]")
{
    pmacc::async::Context context;
    for(bool stop : {false, true})
    {
        caravan::EventSource source;
        auto complete = [&]
        {
            if(stop)
                source.setStopped();
            else
                source.setFailed(std::make_exception_ptr(std::runtime_error("backend failure")));
        };
        if(stop)
            CHECK_THROWS_AS(context.wait(source.event(), complete), caravan::StoppedError);
        else
            CHECK_THROWS_AS(context.wait(source.event(), complete), std::runtime_error);
    }

    caravan::EventSource pending;
    CHECK_THROWS_AS(
        context.wait(pending.event(), [] { throw std::runtime_error("progress failure"); }),
        std::runtime_error);
    pending.setReady(); // The earlier wait's wakeup must not reference its destroyed stack.
    context.runReady();
    context.wait(pending.event());
    // Context destruction also waits on the already-joining scope.
}
