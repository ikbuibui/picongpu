/* Copyright 2025
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_range_equals.hpp"
#include "picongpu/plugins/binning/BinningData.hpp"
#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/FunctorDescription.hpp"
#include "picongpu/plugins/binning/axis/Axis.hpp"
#include "picongpu/plugins/binning/axis/LinearAxis.hpp"
#include "picongpu/plugins/binning/binners/ParticleBinner.hpp"

#include <algorithm>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>


// static_assert(false, "sadfasd");
using namespace picongpu::plugins::binning;


auto getPositionY = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& particle) -> int
{
    auto posBin = getParticlePosition<DomainOrigin::TOTAL>(domainInfo, particle);
    return posBin[1];
};

// Create Functor Description
auto cellPositionYDescription = createFunctorDescription<int>(getPositionY, "position_axisY");

// Create Axis Splitting
auto rangeY = axis::Range{0, 1};
auto cellY_splitting = axis::AxisSplitting(rangeY, 1);

// Create Axis
auto ax_y = axis::createLinear(cellY_splitting, cellPositionYDescription);


auto depData = createFunctorDescription<double>([]() -> double {}, "test");
auto bd = ParticleBinningData("binnerOutputName", std::make_tuple(ax_y), std::tuple<>{}, depData, std::tuple<>{});
// auto binner = make_unique<ParticleBinner>(bd, cellDescription);
