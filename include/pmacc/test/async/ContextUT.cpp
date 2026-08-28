/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <pmacc/async/Context.hpp>

#include <thread>

#include <caravan/core.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("PMacc async context owns work and drives host continuations", "[async]")
{
    pmacc::async::Context context;
    auto const applicationThread = std::this_thread::get_id();
    bool ran = false;

    auto operation = context.spawn(
        caravan::then(
            caravan::asSender(caravan::readyEvent()),
            [&]
            {
                ran = true;
                CHECK(std::this_thread::get_id() == applicationThread);
            }));

    CHECK(operation.state() == caravan::CompletionState::pending);
    context.wait(operation);
    CHECK(ran);

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
