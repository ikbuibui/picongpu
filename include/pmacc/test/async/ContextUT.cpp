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

    caravan::EventSource pending;
    caravan::EventSource checked;
    context.scheduler().post(
        [&]
        {
            CHECK_THROWS_AS(context.wait(pending.event()), std::logic_error);
            checked.setReady();
        });
    context.wait(checked.event());
}
