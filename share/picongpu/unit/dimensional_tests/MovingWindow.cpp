/* Copyright 2025 Tapish Narwal
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/test/PMaccFixture.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <picongpu/defines.hpp>
#include <picongpu/simulation/control/MovingWindow.hpp>

using namespace picongpu;

TEST_CASE("unit::MovingWindow_ComputedConstants", "[movingWindow test]")
{
    SECTION("Conventional move point")
    {
        MovingWindow::ComputedConstants const constants{1000u, 100u, 2.0, 1.0, 0.5};

        CHECK(constants.globalWindowSizeInMoveDirection == 900u);
        CHECK(constants.virtualParticleInitialStartCell == 450);
        CHECK(constants.firstMoveStep == 899);
        CHECK(constants.firstSlideStep == 1099);
        CHECK(constants.virtualParticleGpuLocalCell(1099) == 99u);
        CHECK(constants.virtualParticleGpuLocalCell(1100) == 0u);
        CHECK(constants.movingWindowOriginPositionCells(898) == 0.0);
        CHECK(constants.movingWindowOriginPositionCells(899) == 0.0);
        CHECK(constants.movingWindowOriginPositionCells(900) == 0.5);
    }

    SECTION("Delayed movement uses signed positions")
    {
        MovingWindow::ComputedConstants const constants{1000u, 100u, 2.0, 1.0, 1.5};

        CHECK(constants.virtualParticleInitialStartCell == -450);
        CHECK(constants.firstMoveStep == 2699);
        CHECK(constants.firstSlideStep == 2899);
        CHECK(constants.virtualParticleGpuLocalCell(0) == 50u);
        CHECK(constants.virtualParticleGpuLocalCell(2899) == 99u);
        CHECK(constants.virtualParticleGpuLocalCell(2900) == 0u);
    }

    SECTION("Pre-movement in the hidden GPU row")
    {
        MovingWindow::ComputedConstants const constants{1000u, 100u, 2.0, 1.0, -0.05};

        CHECK(constants.virtualParticleInitialStartCell == 945);
        CHECK(constants.firstMoveStep == -91);
        CHECK(constants.firstSlideStep == 109);
        CHECK(constants.movingWindowOriginPositionCells(0) == 45.5);
        CHECK(constants.virtualParticleGpuLocalCell(109) == 99u);
        CHECK(constants.virtualParticleGpuLocalCell(110) == 0u);
    }

    SECTION("A move point requiring a slide before step zero is invalid")
    {
        CHECK_THROWS_AS((MovingWindow::ComputedConstants{1000u, 100u, 1.0, 1.0, -0.2}), std::runtime_error);
    }
}

//! Helper to setup the PMacc environment
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

struct MovingWindowTestFixture
{
    MovingWindow& window;

    MovingWindowTestFixture() : window(MovingWindow::getInstance())
    {
        window.resetForTesting();
    }

    ~MovingWindowTestFixture()
    {
        window.resetForTesting();
    }
};

TEST_CASE("unit::MovingWindow_origin", "[movingWindow test]")
{
    MovingWindowTestFixture fixture;
    auto& movingWindow = fixture.window;

    pmaccFixture.initGrids(
        pmacc::DataSpace<3>{500, 1000, 700}.shrink<simDim>(),
        pmacc::DataSpace<3>{50, 100, 70}.shrink<simDim>(),
        pmacc::DataSpace<3>{0, 0, 0}.shrink<simDim>());

    uint32_t const globalWindowSizeInMoveDirection = 900u;
    uint32_t const gpuNumCellsInMoveDirection = 100u;

    auto const c = static_cast<float_64>(sim.pic.getSpeedOfLight());
    auto const dt = static_cast<float_64>(sim.pic.getDt());
    // should both be 1 for correctness of the tests
    REQUIRE(c == 1.);
    REQUIRE(dt == 1.);
    // We now assume in tests that c and dt are 1

    auto const lightDistancePerStep = c * dt;

    auto const cellSize = sim.pic.getCellSize();
    constexpr auto moveDirection = MovingWindow::moveDirection;
    auto const cellSizeInMoveDirection = static_cast<float_64>(cellSize[moveDirection]);
    auto const cellsPerStep = lightDistancePerStep / cellSizeInMoveDirection;
    auto const constantsFor = [=](float_64 const movePoint)
    {
        return MovingWindow::ComputedConstants{
            globalWindowSizeInMoveDirection + gpuNumCellsInMoveDirection,
            gpuNumCellsInMoveDirection,
            cellSizeInMoveDirection,
            lightDistancePerStep,
            movePoint};
    };
    auto const checkMovingState
        = [&](MovingWindow::ComputedConstants const& constants, uint32_t const step, uint32_t const expectedSlides)
    {
        auto const origin = movingWindow.getMovingWindowOriginPositionCells(step);
        auto const window = movingWindow.getWindow(step);
        auto const slides = movingWindow.getSlideCounter(step);

        CHECK(std::isfinite(origin[moveDirection]));
        CHECK(
            origin[moveDirection]
            == Catch::Approx(constants.movingWindowOriginPositionCells(static_cast<int64_t>(step))));
        CHECK(
            window.globalDimensions.offset[moveDirection]
            == constants.virtualParticleGpuLocalCell(static_cast<int64_t>(step) + 1));
        CHECK(slides == expectedSlides);

        auto const windowOrigin = static_cast<float_64>(window.globalDimensions.offset[moveDirection])
                                  + static_cast<float_64>(slides * gpuNumCellsInMoveDirection);
        CHECK(origin[moveDirection] >= windowOrigin);
        CHECK(origin[moveDirection] < windowOrigin + 1.0);
    };

    SECTION("Window disabled")
    {
        movingWindow.setEndSlideOnStep(0);
        REQUIRE_FALSE(movingWindow.isEnabled());
        uint32_t const testStep = 100;
        auto origin = movingWindow.getMovingWindowOriginPositionCells(testStep);
        for(unsigned i = 0; i < simDim; ++i)
        {
            CHECK(origin[i] == 0.0);
        }
    }

    SECTION("Window enabled")
    {
        float_64 const movePoint = GENERATE(0.4, 0.8);
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const constants = constantsFor(movePoint);
        auto const firstMoveStep = constants.firstMoveStep;

        SECTION("Window enabled, but not yet moving")
        {
            // Move point is > 0 so we dont immediately start moving.
            REQUIRE(firstMoveStep > 0);

            // At the step just before movement starts, origin must be zero.
            auto origin = movingWindow.getMovingWindowOriginPositionCells(firstMoveStep - 1);
            for(unsigned i = 0; i < simDim; ++i)
            {
                CHECK(origin[i] == 0.0);
            }
        }

        SECTION("Window enabled, and moving")
        {
            uint32_t const testStep = static_cast<uint32_t>(firstMoveStep + 100);
            auto const origin = movingWindow.getMovingWindowOriginPositionCells(testStep);

            auto const expectedOrigin = constants.movingWindowOriginPositionCells(testStep);

            for(unsigned i = 0; i < simDim; ++i)
            {
                if(i == moveDirection)
                {
                    CHECK(origin[i] == Catch::Approx(expectedOrigin));
                }
                else
                {
                    CHECK(origin[i] == 0.0);
                }
            }
        }
    }

    SECTION("Delayed movement outside the conventional move-point interval")
    {
        float_64 const movePoint = 1.5;
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);

        auto const constants = constantsFor(movePoint);
        auto const firstMoveStep = constants.firstMoveStep;
        auto const firstSlideStep = constants.firstSlideStep;
        REQUIRE(firstMoveStep > 0);
        REQUIRE(firstSlideStep > firstMoveStep);

        uint32_t const beforeFirstMoveStep = static_cast<uint32_t>(firstMoveStep - 1);
        auto const originBeforeFirstMove = movingWindow.getMovingWindowOriginPositionCells(beforeFirstMoveStep);
        auto const windowBeforeFirstMove = movingWindow.getWindow(beforeFirstMoveStep);
        CHECK(originBeforeFirstMove[moveDirection] == 0.0);
        CHECK(windowBeforeFirstMove.globalDimensions.offset[moveDirection] == 0u);
        CHECK(movingWindow.getSlideCounter(beforeFirstMoveStep) == 0u);
        CHECK_FALSE(movingWindow.slideInCurrentStep(beforeFirstMoveStep));

        uint32_t const firstMoveStepAsUint = static_cast<uint32_t>(firstMoveStep);
        checkMovingState(constants, firstMoveStepAsUint, 0u);
        checkMovingState(constants, firstMoveStepAsUint + 1u, 0u);

        uint32_t const firstSlideStepAsUint = static_cast<uint32_t>(firstSlideStep);
        CHECK_FALSE(movingWindow.slideInCurrentStep(firstSlideStepAsUint - 1u));
        CHECK(movingWindow.slideInCurrentStep(firstSlideStepAsUint));
        checkMovingState(constants, firstSlideStepAsUint, 1u);
    }

    SECTION("Negative move point supports pre-movement in the hidden GPU row")
    {
        float_64 const movePoint = -0.05;
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);

        auto const constants = constantsFor(movePoint);
        auto const firstMoveStep = constants.firstMoveStep;
        auto const firstSlideStep = constants.firstSlideStep;
        REQUIRE(firstMoveStep < 0);
        REQUIRE(firstSlideStep > 0);

        checkMovingState(constants, 0u, 0u);

        uint32_t const firstSlideStepAsUint = static_cast<uint32_t>(firstSlideStep);
        CHECK_FALSE(movingWindow.slideInCurrentStep(firstSlideStepAsUint - 1u));
        CHECK(movingWindow.slideInCurrentStep(firstSlideStepAsUint));
        checkMovingState(constants, firstSlideStepAsUint, 1u);
    }

    SECTION("Negative move point requiring pre-zero slides is rejected")
    {
        movingWindow.setMovePoint(-0.2);
        movingWindow.setEndSlideOnStep(10000);
        CHECK_THROWS_AS(movingWindow.getWindow(0), std::runtime_error);
    }

    SECTION("Window instantly moving")
    {
        float_64 const movePoint = 0.0;
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const origin0 = movingWindow.getMovingWindowOriginPositionCells(0);
        // The comoving particle origin is defined to start at from the location equal to the distance travelled from
        // the origin at c in one timestep at first move step3
        CHECK(origin0[moveDirection] == Catch::Approx(cellsPerStep));
    }

    SECTION("Window position evolution")
    {
        float_64 const movePoint = GENERATE(0.0, 0.341, 0.6, 1.5);
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const constants = constantsFor(movePoint);
        auto previousOrigin = movingWindow.getMovingWindowOriginPositionCells(0)[moveDirection];

        for(uint32_t step = 1; step < endStep; ++step)
        {
            auto const origin = movingWindow.getMovingWindowOriginPositionCells(step);
            auto const signedStep = static_cast<int64_t>(step);

            if(signedStep < constants.firstMoveStep)
                CHECK(origin[moveDirection] == 0.0);
            else if(signedStep > constants.firstMoveStep)
                CHECK(origin[moveDirection] - previousOrigin == Catch::Approx(cellsPerStep));

            auto const window = movingWindow.getWindow(step);
            auto const windowRelativeParticlePos
                = origin[moveDirection]
                  - (window.globalDimensions.offset[moveDirection]
                     + movingWindow.getSlideCounter(step) * gpuNumCellsInMoveDirection);

            if(signedStep < constants.firstMoveStep)
                CHECK(windowRelativeParticlePos == 0.0);
            else
                CHECK((windowRelativeParticlePos >= 0.0 && windowRelativeParticlePos < 1.0));

            previousOrigin = origin[moveDirection];
        }
    }


    SECTION("Origin is clamped after endSlidingOnStep")
    {
        uint32_t const endStep = 500;
        movingWindow.setMovePoint(0.0);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const originAtEnd = movingWindow.getMovingWindowOriginPositionCells(endStep);
        auto const originAfterEnd = movingWindow.getMovingWindowOriginPositionCells(endStep + 100);

        for(unsigned i = 0; i < simDim; ++i)
        {
            CHECK(originAfterEnd[i] == Catch::Approx(originAtEnd[i]));
        }
    }
}
