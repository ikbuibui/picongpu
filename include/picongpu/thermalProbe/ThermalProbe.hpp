/* Copyright 2026 PIConGPU contributors
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
 * along with PIConGPU. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

/** Test-only observability hook for the Thermal Caravan migration slice (WP6/WP0).
 *
 * This header is compiled only when `PICONGPU_THERMAL_PROBE` is defined. It is
 * deliberately not a plugin and does not register anything with the plugin
 * connector. It measures a fixed, backend-independent set of diagnostics on the
 * control thread and writes them to per-rank files:
 *
 *   - `<dir>/thermal_probe_rank<rank>.csv`     one row per species and observation
 *   - `<dir>/thermal_snapshot_rank<rank>_step<step>.csv`  owned E/B/J cells
 *
 * The runtime behaviour is controlled by environment variables:
 *
 *   - `PICONGPU_THERMAL_PROBE_DIR`      output directory; unset disables output
 *   - `PICONGPU_THERMAL_PROBE_STEPS`    comma separated step list, or `final`;
 *                                       default: every observation
 *   - `PICONGPU_THERMAL_PROBE_SNAPSHOTS` `0` disables the full field snapshots
 *
 * The same header is used by the legacy pre-migration checkout. Define
 * `PICONGPU_THERMAL_PROBE_LEGACY` there so the scheduling wrappers use the
 * synchronous event-system API instead of Caravan senders. The kernel, the
 * field-statistics code, and the output schema are shared verbatim so the two
 * trees produce directly comparable records.
 */

#include "picongpu/algorithms/KinEnergy.hpp"
#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"
#include "picongpu/fields/FieldJ.hpp"
#include "picongpu/particles/Particles.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/simulation/GridController.hpp>
#include <pmacc/memory/buffers/GridBuffer.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/simulationControl/SimulationDescription.hpp>

#if !defined(PICONGPU_THERMAL_PROBE_LEGACY)
#    include <caravan/alpaka.hpp>
#    include <caravan/core.hpp>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace picongpu::thermalProbe
{
    /** Number of particle-reduction slots written by the device kernel. */
    enum ReductionSlot : uint32_t
    {
        R_Count = 0,
        R_Weight,
        R_MomX,
        R_MomY,
        R_MomZ,
        R_Ekin,
        R_Etot,
        R_Nonfinite,
        R_Num
    };

    using ReductionBuffer = pmacc::GridBuffer<float_64, DIM1>;

    /** Device-side finiteness test that works on host and accelerator.
     *
     * Uses only comparisons so it does not rely on a backend `isfinite`
     * overload. NaN fails `v == v`; infinity makes `v - v` NaN.
     */
    template<typename T>
    HDINLINE bool componentFinite(T const value)
    {
        return (value == value) && ((value - value) == T(0));
    }

    /** Reduce the owned macro-particle state of one species.
     *
     * The area is `CORE + BORDER` so guards are excluded and a global sum over
     * ranks counts every owned macro particle exactly once. Momentum and
     * energy use the stored (already macro-weighted) momentum, matching
     * `EnergyParticles.x.cpp`.
     */
    struct KernelThermalParticleProbe
    {
        template<typename T_ParBox, typename T_Mapping, typename T_Worker>
        DINLINE void operator()(T_Worker const& worker, T_ParBox pb, float_64* gRed, T_Mapping const mapper) const
        {
            PMACC_SMEM(worker, shCount, uint64_cu);
            PMACC_SMEM(worker, shWeight, float_X);
            PMACC_SMEM(worker, shMomX, float_X);
            PMACC_SMEM(worker, shMomY, float_X);
            PMACC_SMEM(worker, shMomZ, float_X);
            PMACC_SMEM(worker, shEkin, float_X);
            PMACC_SMEM(worker, shEtot, float_X);
            PMACC_SMEM(worker, shBad, uint64_cu);

            float_X localWeight = float_X(0.0);
            float_X localMomX = float_X(0.0);
            float_X localMomY = float_X(0.0);
            float_X localMomZ = float_X(0.0);
            float_X localEkin = float_X(0.0);
            float_X localEtot = float_X(0.0);
            uint64_cu localCount = uint64_cu(0);
            uint64_cu localBad = uint64_cu(0);

            auto masterOnly = lockstep::makeMaster(worker);
            masterOnly(
                [&]()
                {
                    shCount = uint64_cu(0);
                    shWeight = float_X(0.0);
                    shMomX = float_X(0.0);
                    shMomY = float_X(0.0);
                    shMomZ = float_X(0.0);
                    shEkin = float_X(0.0);
                    shEtot = float_X(0.0);
                    shBad = uint64_cu(0);
                });
            worker.sync();

            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(DataSpace<simDim>(worker.blockDomIdxND())));

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, pb, superCellIdx);
            if(!forEachParticle.hasParticles())
                return;

            forEachParticle(
                [&](auto const&, auto& particle)
                {
                    float3_X const pos = particle[position_];
                    float3_X const mom = particle[momentum_];
                    float_X const weighting = particle[weighting_];

                    bool const finite = componentFinite(pos.x()) && componentFinite(pos.y())
                                        && componentFinite(pos.z()) && componentFinite(mom.x())
                                        && componentFinite(mom.y()) && componentFinite(mom.z())
                                        && componentFinite(weighting);

                    localCount += uint64_cu(1);
                    if(!finite)
                        localBad += uint64_cu(1);

                    localWeight += weighting;
                    localMomX += mom.x();
                    localMomY += mom.y();
                    localMomZ += mom.z();

                    float_X const mass = picongpu::traits::attribute::getMass(weighting, particle);
                    localEkin += KinEnergy<>()(mom, mass);

                    float_X const mom2 = pmacc::math::l2norm2(mom);
                    float_X const c = sim.pic.getSpeedOfLight();
                    localEtot += pmacc::math::sqrt(mom2 + mass * mass * c * c) * c;
                });

            alpaka::atomicAdd(worker.getAcc(), &shCount, localCount, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shBad, localBad, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shWeight, localWeight, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shMomX, localMomX, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shMomY, localMomY, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shMomZ, localMomZ, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shEkin, localEkin, ::alpaka::hierarchy::Threads{});
            alpaka::atomicAdd(worker.getAcc(), &shEtot, localEtot, ::alpaka::hierarchy::Threads{});

            worker.sync();

            masterOnly(
                [&]()
                {
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_Count],
                        static_cast<float_64>(shCount),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_Nonfinite],
                        static_cast<float_64>(shBad),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_Weight],
                        static_cast<float_64>(shWeight),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_MomX],
                        static_cast<float_64>(shMomX),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_MomY],
                        static_cast<float_64>(shMomY),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_MomZ],
                        static_cast<float_64>(shMomZ),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_Ekin],
                        static_cast<float_64>(shEkin),
                        ::alpaka::hierarchy::Blocks{});
                    alpaka::atomicAdd(
                        worker.getAcc(),
                        &gRed[R_Etot],
                        static_cast<float_64>(shEtot),
                        ::alpaka::hierarchy::Blocks{});
                });
        }
    };

    //! Runtime configuration parsed from the environment.
    struct Config
    {
        bool enabled = false;
        bool snapshots = true;
        bool finalOnly = false;
        bool allSteps = true;
        std::filesystem::path directory;
        std::vector<uint32_t> steps;
    };

    inline Config const& config()
    {
        static Config const instance = []
        {
            Config c;
            char const* directory = std::getenv("PICONGPU_THERMAL_PROBE_DIR");
            if(directory == nullptr || *directory == '\0')
                return c;
            c.enabled = true;
            c.directory = directory;

            char const* snapshots = std::getenv("PICONGPU_THERMAL_PROBE_SNAPSHOTS");
            if(snapshots != nullptr && std::string(snapshots) == "0")
                c.snapshots = false;

            char const* steps = std::getenv("PICONGPU_THERMAL_PROBE_STEPS");
            if(steps == nullptr || *steps == '\0')
                return c;

            std::string const spec(steps);
            if(spec == "final")
            {
                c.finalOnly = true;
                return c;
            }

            c.allSteps = false;
            std::stringstream stream(spec);
            std::string token;
            while(std::getline(stream, token, ','))
            {
                if(!token.empty())
                    c.steps.push_back(static_cast<uint32_t>(std::stoul(token)));
            }
            return c;
        }();
        return instance;
    }

    inline bool probeEnabled()
    {
        return config().enabled;
    }

    inline uint32_t currentRank()
    {
        return pmacc::Environment<simDim>::get().GridController().getGlobalRank();
    }

    inline bool observeStep(uint32_t const step)
    {
        Config const& c = config();
        if(!c.enabled)
            return false;
        if(c.finalOnly)
        {
            uint32_t const runSteps = pmacc::Environment<>::get().SimulationDescription().getRunSteps();
            return step == runSteps;
        }
        if(c.allSteps)
            return true;
        return std::find(c.steps.begin(), c.steps.end(), step) != c.steps.end();
    }

    //! Particle reduction read back from the device.
    struct ParticleStats
    {
        uint64_t count = 0;
        double weightSum = 0.0;
        double momX = 0.0;
        double momY = 0.0;
        double momZ = 0.0;
        double ekin = 0.0;
        double etot = 0.0;
        uint64_t nonfinite = 0;
    };

    //! Field statistics over the owned (non-guard) cells of one rank.
    struct FieldStats
    {
        bool available = false;
        double energy = 0.0;
        double absSum = 0.0;
        double min = std::numeric_limits<double>::infinity();
        double max = -std::numeric_limits<double>::infinity();
        uint64_t nonfinite = 0;
    };

    template<typename T_Vector>
    inline void accumulateField(T_Vector const& value, FieldStats& stats)
    {
        stats.available = true;
        for(int d = 0; d < T_Vector::dim; ++d)
        {
            double const component = static_cast<double>(value[d]);
            if(!std::isfinite(component))
            {
                stats.nonfinite += 1;
                continue;
            }
            stats.energy += component * component;
            stats.absSum += std::abs(component);
            stats.min = std::min(stats.min, component);
            stats.max = std::max(stats.max, component);
        }
    }

    inline DataSpace<simDim> linearToNd(uint64_t linear, DataSpace<simDim> const& size)
    {
        DataSpace<simDim> index;
        for(uint32_t d = 0; d < simDim; ++d)
        {
            index[d] = static_cast<int>(linear % static_cast<uint64_t>(size[d]));
            linear /= static_cast<uint64_t>(size[d]);
        }
        return index;
    }

    inline void writeVec3(std::ostream& stream, float3_X const& value)
    {
        stream << std::setprecision(std::numeric_limits<float_X>::max_digits10) << value.x() << ',' << value.y() << ','
               << value.z();
    }

    /** Collect owned E/B/J statistics and, when enabled, write the owned cells.
     *
     * Guards are excluded from every statistic. The snapshot carries global
     * cell coordinates so per-rank files can be merged offline. The J columns
     * are `nan` when @p fieldJ is null (initialization has no defined current).
     */
    inline void collectAndWrite(
        std::shared_ptr<FieldE> const& fieldE,
        std::shared_ptr<FieldB> const& fieldB,
        std::shared_ptr<FieldJ> const& fieldJ,
        uint32_t const step,
        FieldStats& statsE,
        FieldStats& statsB,
        FieldStats& statsJ)
    {
        auto const layout = fieldE->getGridLayout();
        DataSpace<simDim> const size = layout.sizeWithoutGuardND();
        DataSpace<simDim> const guard = layout.guardSizeND();
        DataSpace<simDim> const localOffset = pmacc::Environment<simDim>::get().SubGrid().getLocalDomain().offset;

        auto const boxE = fieldE->getHostDataBox();
        auto const boxB = fieldB->getHostDataBox();
        bool const jAvailable = (fieldJ != nullptr);
        auto const boxJ = jAvailable ? fieldJ->getHostDataBox() : fieldE->getHostDataBox();

        std::ofstream snapshot;
        if(config().snapshots)
        {
            snapshot.open(
                config().directory
                    / ("thermal_snapshot_rank" + std::to_string(currentRank()) + "_step" + std::to_string(step)
                       + ".csv"),
                std::ios::out | std::ios::trunc);
            if(!snapshot)
                throw std::runtime_error("thermal probe: cannot open snapshot file in " + config().directory.string());
            snapshot << "gx,gy,gz,Ex,Ey,Ez,Bx,By,Bz,Jx,Jy,Jz\n";
        }

        uint64_t const cells = static_cast<uint64_t>(size.productOfComponents());
        for(uint64_t linear = 0; linear < cells; ++linear)
        {
            DataSpace<simDim> const index = linearToNd(linear, size);
            DataSpace<simDim> const guarded = index + guard;

            float3_X const valueE = boxE(guarded);
            float3_X const valueB = boxB(guarded);
            accumulateField(valueE, statsE);
            accumulateField(valueB, statsB);

            float3_X valueJ = float3_X(float_X(0.0));
            if(jAvailable)
            {
                valueJ = boxJ(guarded);
                accumulateField(valueJ, statsJ);
            }

            if(snapshot)
            {
                DataSpace<simDim> const global = index + localOffset;
                snapshot << global[0] << ',' << global[1] << ',' << global[2] << ',';
                writeVec3(snapshot, valueE);
                snapshot << ',';
                writeVec3(snapshot, valueB);
                snapshot << ',';
                if(jAvailable)
                    writeVec3(snapshot, valueJ);
                else
                    snapshot << "nan,nan,nan";
                snapshot << '\n';
            }
        }
    }

    inline void appendSummary(
        uint32_t const step,
        std::string const& stage,
        std::string const& species,
        ParticleStats const& particle,
        FieldStats const& statsE,
        FieldStats const& statsB,
        FieldStats const& statsJ)
    {
        std::filesystem::path const path
            = config().directory / ("thermal_probe_rank" + std::to_string(currentRank()) + ".csv");
        bool const writeHeader = !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0u;

        std::ofstream out(path, std::ios::out | std::ios::app);
        if(!out)
            throw std::runtime_error("thermal probe: cannot open summary file " + path.string());
        if(writeHeader)
            out << "step,stage,species,count,weight_sum,mom_x,mom_y,mom_z,ekin,etot,particle_nonfinite,"
                   "E_energy,E_abs,E_min,E_max,E_nonfinite,B_energy,B_abs,B_min,B_max,B_nonfinite,"
                   "J_available,J_energy,J_abs,J_min,J_max,J_nonfinite\n";

        out << std::setprecision(std::numeric_limits<double>::max_digits10) << step << ',' << stage << ',' << species
            << ',' << particle.count << ',' << particle.weightSum << ',' << particle.momX << ',' << particle.momY
            << ',' << particle.momZ << ',' << particle.ekin << ',' << particle.etot << ',' << particle.nonfinite << ','
            << statsE.energy << ',' << statsE.absSum << ',' << statsE.min << ',' << statsE.max << ','
            << statsE.nonfinite << ',' << statsB.energy << ',' << statsB.absSum << ',' << statsB.min << ','
            << statsB.max << ',' << statsB.nonfinite << ',' << (statsJ.available ? 1 : 0) << ',' << statsJ.energy
            << ',' << statsJ.absSum << ',' << statsJ.min << ',' << statsJ.max << ',' << statsJ.nonfinite << '\n';
    }

    //! Aggregated per-species reduction plus its name.
    struct SpeciesReduction
    {
        std::string name;
        std::shared_ptr<ReductionBuffer> buffer;
    };

    inline ParticleStats readReduction(ReductionBuffer const& buffer)
    {
        auto const host = buffer.getHostBuffer().getDataBox();
        ParticleStats stats;
        stats.count = static_cast<uint64_t>(host[DataSpace<DIM1>(R_Count)]);
        stats.weightSum = static_cast<double>(host[DataSpace<DIM1>(R_Weight)]);
        stats.momX = static_cast<double>(host[DataSpace<DIM1>(R_MomX)]);
        stats.momY = static_cast<double>(host[DataSpace<DIM1>(R_MomY)]);
        stats.momZ = static_cast<double>(host[DataSpace<DIM1>(R_MomZ)]);
        stats.ekin = static_cast<double>(host[DataSpace<DIM1>(R_Ekin)]);
        stats.etot = static_cast<double>(host[DataSpace<DIM1>(R_Etot)]);
        stats.nonfinite = static_cast<uint64_t>(host[DataSpace<DIM1>(R_Nonfinite)]);
        return stats;
    }

#if defined(PICONGPU_THERMAL_PROBE_LEGACY)

    /** Launch the reduction kernel for one species and wait for its host copy. */
    template<typename T_SpeciesType>
    void runSpeciesReduction(MappingDesc* cellDescription, std::vector<SpeciesReduction>& reductions)
    {
        using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
        using FrameType = typename SpeciesType::FrameType;

        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
        auto species = dc.get<SpeciesType>(FrameType::getName());
        if(species == nullptr)
            return;

        auto buffer = std::make_shared<ReductionBuffer>(DataSpace<DIM1>(R_Num));
        buffer->getDeviceBuffer().setValue(float_64(0.0));

        auto const mapper = makeAreaMapper<CORE + BORDER>(*cellDescription);
        PMACC_LOCKSTEP_KERNEL(KernelThermalParticleProbe{})
            .config(
                mapper.getGridDim(),
                *species)(species->getDeviceParticlesBox(), buffer->getDeviceBuffer().data(), mapper);
        buffer->deviceToHost();

        reductions.push_back({FrameType::getName(), std::move(buffer)});
    }

    template<typename... TSpecies>
    void runSpeciesReductionsImpl(
        MappingDesc* cellDescription,
        std::vector<SpeciesReduction>& reductions,
        pmacc::mp_list<TSpecies...>)
    {
        (runSpeciesReduction<TSpecies>(cellDescription, reductions), ...);
    }

    /** Observe one completed simulation state using the legacy synchronous API.
     *
     * `includeJ` is false during initialization because the current is not
     * defined until the first `CurrentReset` + deposition. All field
     * synchronizations and the reduction host copies are completed before any
     * host value is read.
     */
    inline void runProbe(
        MappingDesc* cellDescription,
        uint32_t const step,
        std::string const& stage,
        bool const includeJ)
    {
        if(!probeEnabled() || !observeStep(step))
            return;

        std::vector<SpeciesReduction> reductions;
        runSpeciesReductionsImpl(cellDescription, reductions, VectorAllSpecies{});

        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
        auto fieldE = dc.get<FieldE>(FieldE::getName());
        auto fieldB = dc.get<FieldB>(FieldB::getName());
        fieldE->synchronize();
        fieldB->synchronize();

        std::shared_ptr<FieldJ> fieldJ;
        if(includeJ)
        {
            fieldJ = dc.get<FieldJ>(FieldJ::getName());
            fieldJ->synchronize();
        }

        FieldStats statsE;
        FieldStats statsB;
        FieldStats statsJ;
        collectAndWrite(fieldE, fieldB, fieldJ, step, statsE, statsB, statsJ);

        for(auto const& reduction : reductions)
            appendSummary(step, stage, reduction.name, readReduction(*reduction.buffer), statsE, statsB, statsJ);
    }

#else // PICONGPU_THERMAL_PROBE_LEGACY

    /** Start the reduction kernel for one species after @p previous, without waiting. */
    template<typename T_SpeciesType>
    void startSpeciesReduction(
        caravan::ControlContext& context,
        MappingDesc* cellDescription,
        std::vector<SpeciesReduction>& reductions,
        std::vector<caravan::Event>& events,
        caravan::Event previous)
    {
        using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
        using FrameType = typename SpeciesType::FrameType;

        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
        auto species = dc.get<SpeciesType>(FrameType::getName());
        if(species == nullptr)
            return;

        auto buffer = std::make_shared<ReductionBuffer>(DataSpace<DIM1>(R_Num));
        auto initialize = caravan::alpaka::fill(buffer->getDeviceBuffer().getOwnedAlpakaView(), float_64(0.0));
        auto const mapper = makeAreaMapper<CORE + BORDER>(*cellDescription);
        auto probeKernel = PMACC_LOCKSTEP_KERNEL(KernelThermalParticleProbe{})
                               .config(mapper.getGridDim(), *species)(
                                   species->getDeviceParticlesBox(),
                                   caravan::retain(
                                       buffer->getDeviceBuffer().data(),
                                       buffer->getDeviceBuffer().getOwnedAlpakaView()),
                                   mapper);
        auto copy = buffer->deviceToHost();
        auto& device = pmacc::Environment<>::get().DeviceContext();
        events.push_back(context.spawn(
            caravan::alpaka::withDevice(
                device,
                caravan::asSender(std::move(previous)) | caravan::sequence(std::move(initialize))
                    | caravan::sequence(std::move(probeKernel)) | caravan::sequence(std::move(copy)))));
        reductions.push_back({FrameType::getName(), std::move(buffer)});
    }

    template<typename... TSpecies>
    void startSpeciesReductionsImpl(
        caravan::ControlContext& context,
        MappingDesc* cellDescription,
        std::vector<SpeciesReduction>& reductions,
        std::vector<caravan::Event>& events,
        caravan::Event previous,
        pmacc::mp_list<TSpecies...>)
    {
        (startSpeciesReduction<TSpecies>(context, cellDescription, reductions, events, previous), ...);
    }

    /** Observe one completed simulation state using Caravan operations.
     *
     * Every reduction/readback is started after @p previous and explicitly
     * awaited before host values are inspected. The reduction buffers and
     * fields outlive the wait through the local shared pointers.
     */
    inline void runProbe(
        caravan::ControlContext& context,
        MappingDesc* cellDescription,
        uint32_t const step,
        std::string const& stage,
        bool const includeJ,
        caravan::Event previous)
    {
        if(!probeEnabled() || !observeStep(step))
            return;

        std::vector<SpeciesReduction> reductions;
        std::vector<caravan::Event> events;
        startSpeciesReductionsImpl(context, cellDescription, reductions, events, previous, VectorAllSpecies{});

        pmacc::DataConnector& dc = pmacc::Environment<>::get().DataConnector();
        auto fieldE = dc.get<FieldE>(FieldE::getName());
        auto fieldB = dc.get<FieldB>(FieldB::getName());
        events.push_back(fieldE->synchronize(context, previous));
        events.push_back(fieldB->synchronize(context, previous));

        std::shared_ptr<FieldJ> fieldJ;
        if(includeJ)
        {
            fieldJ = dc.get<FieldJ>(FieldJ::getName());
            events.push_back(fieldJ->synchronize(context, previous));
        }

        context.wait(caravan::whenAll(std::span<caravan::Event const>{events.data(), events.size()}));

        FieldStats statsE;
        FieldStats statsB;
        FieldStats statsJ;
        collectAndWrite(fieldE, fieldB, fieldJ, step, statsE, statsB, statsJ);

        for(auto const& reduction : reductions)
            appendSummary(step, stage, reduction.name, readReduction(*reduction.buffer), statsE, statsB, statsJ);
    }

#endif // PICONGPU_THERMAL_PROBE_LEGACY

} // namespace picongpu::thermalProbe
