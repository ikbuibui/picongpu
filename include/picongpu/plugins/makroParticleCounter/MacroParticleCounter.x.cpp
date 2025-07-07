/* Copyright 2013-2024 Axel Huebl, Heiko Burau, Rene Widera, Tapish Narwal
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


#if (ENABLE_OPENPMD == 1)
#    include "picongpu/defines.hpp"
#    include "picongpu/particles/filter/filter.hpp"
#    include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"
#    include "picongpu/plugins/PhaseSpace/AxisDescription.hpp"
#    include "picongpu/plugins/PhaseSpace/DumpHBufferOpenPMD.hpp"
#    include "picongpu/plugins/PhaseSpace/Pair.hpp"
#    include "picongpu/plugins/PhaseSpace/PhaseSpaceFunctors.hpp"
#    include "picongpu/plugins/PluginRegistry.hpp"
#    include "picongpu/plugins/common/openPMDAttributes.hpp"
#    include "picongpu/plugins/common/openPMDDefaultExtension.hpp"
#    include "picongpu/plugins/common/openPMDWriteMeta.hpp"
#    include "picongpu/plugins/misc/misc.hpp"
#    include "picongpu/plugins/multi/multi.hpp"
#    include "picongpu/traits/frame/GetMass.hpp"

#    include <pmacc/communication/manager_common.hpp>
#    include <pmacc/lockstep/lockstep.hpp>
#    include <pmacc/mappings/simulation/Filesystem.hpp>
#    include <pmacc/math/Vector.hpp>
#    include <pmacc/memory/buffers/GridBuffer.hpp>
#    include <pmacc/mpi/MPIReduce.hpp>
#    include <pmacc/mpi/reduceMethods/Reduce.hpp>
#    include <pmacc/pluginSystem/INotify.hpp>
#    include <pmacc/traits/HasFlag.hpp>
#    include <pmacc/traits/HasIdentifiers.hpp>

#    include <memory>
#    include <string>
#    include <utility>

#    include <mpi.h>

namespace picongpu
{
    using namespace pmacc;

    //! Count macro particles per superCell
    struct KernelCountMacroParticles
    {
        template<typename ParBox, typename CounterBox, typename Mapping, typename T_Worker>
        DINLINE void operator()(
            T_Worker const& worker,
            ParBox parBox,
            CounterBox counterBox,
            Mapping mapper,
            auto filter) const
        {
            DataSpace<simDim> const superCellIdx(mapper.getSuperCellIndex(worker.blockDomIdxND()));
            /* counterBox has no guarding supercells*/
            DataSpace<simDim> const superCellIdxNoGuard = superCellIdx - mapper.getGuardingSuperCells();

            PMACC_SMEM(worker, counterValue, uint64_cu);

            auto masterOnly = lockstep::makeMaster(worker);

            masterOnly([&]() { counterValue = 0; });
            worker.sync();

            auto accFilter = filter(worker, superCellIdxNoGuard);

            auto forEachParticle = pmacc::particles::algorithm::acc::makeForEach(worker, parBox, superCellIdx);

            // end kernel if we have no particles
            if(!forEachParticle.hasParticles())
                return;

            forEachParticle(
                [&accFilter, &counterValue](auto const& lockstepWorker, auto& particle)
                {
                    if(accFilter(lockstepWorker, particle))
                    {
                        alpaka::atomicAdd(
                            lockstepWorker.getAcc(),
                            &counterValue,
                            static_cast<uint64_cu>(1LU),
                            ::alpaka::hierarchy::Threads{});
                    }
                });

            worker.sync();

            masterOnly(
                [&]()
                {
                    PMACC_DEVICE_ASSERT_MSG(
                        counterValue == forEachParticle.numParticles(),
                        "[macroParticlesCounter] Number of particles counted and given by the iteration algorithm "
                        "differ.");
                    counterBox(superCellIdxNoGuard) = counterValue;
                });
        }
    };

    namespace po = boost::program_options;

    /** Count macro particle of a species and write down the result to a global HDF5 file.
     *
     * - count the total number of macro particles per supercell
     * - store one number (size_t) per supercell in a mesh
     * - Output: - create a folder with the name of the plugin
     *           - per time step one file with the name "result_[currentStep].h5" is created
     *             (or a different extension in case of another openPMD backend)
     *           - the attribute name in the openPMD file is "makroParticlePerSupercell"
     *
     */
    template<typename T_Species>
    class MacroParticleCounter : public plugins::multi::IInstance
    {
    public:
        using Species = T_Species;

        struct Help : public plugins::multi::IHelp
        {
            /** creates an instance
             *
             * @param help plugin defined help
             * @param id index of the plugin, range: [0;help->getNumPlugins())
             */
            std::shared_ptr<IInstance> create(
                std::shared_ptr<IHelp>& help,
                size_t const id,
                MappingDesc* cellDescription) override
            {
                return std::make_shared<MacroParticleCounter<Species>>(help, id, cellDescription);
            }

            // find all valid filter for the current used species
            template<typename T>
            using Op = typename particles::traits::GenerateSolversIfSpeciesEligible<T, Species>::type;
            using EligibleFilters = pmacc::mp_flatten<pmacc::mp_transform<Op, particles::filter::AllParticleFilters>>;

            //! periodicity of computing the particle energy
            plugins::multi::Option<std::string> notifyPeriod = {"period", "notify period"};
            plugins::multi::Option<std::string> filter = {"filter", "particle filter: "};

            plugins::multi::Option<std::string> file_name_extension
                = {"ext",
                   "openPMD filename extension (this controls the"
                   "backend picked by the openPMD API)",
                   openPMD::getDefaultExtension().c_str()};

            plugins::multi::Option<std::string> file_name_infix
                = {"infix",
                   "openPMD filename infix (default: '_%06T' for file-based iteration layout, pick 'NULL' for "
                   "group-based layout",
                   std::string("_%06T").c_str()};

            plugins::multi::Option<std::string> json_config
                = {"json", "advanced (backend) configuration for openPMD in JSON format", "{}"};

            //! string list with all possible particle filters
            std::string concatenatedFilterNames;
            std::vector<std::string> allowedFilters;

            ///! method used by plugin controller to get --help description
            void registerHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
                meta::ForEach<EligibleFilters, plugins::misc::AppendName<boost::mpl::_1>> getEligibleFilterNames;
                getEligibleFilterNames(allowedFilters);

                concatenatedFilterNames = plugins::misc::concatenateToString(allowedFilters, ", ");

                notifyPeriod.registerHelp(desc, masterPrefix + prefix);
                filter.registerHelp(desc, masterPrefix + prefix, std::string("[") + concatenatedFilterNames + "]");

                file_name_extension.registerHelp(desc, masterPrefix + prefix);
                file_name_infix.registerHelp(desc, masterPrefix + prefix);

                json_config.registerHelp(desc, masterPrefix + prefix);
            }

            void expandHelp(
                boost::program_options::options_description& desc,
                std::string const& masterPrefix = std::string{}) override
            {
            }

            void validateOptions() override
            {
                if(notifyPeriod.size() != filter.size())
                    throw std::runtime_error(
                        name + ": parameter filter and period are not used the same number of times");

                // check if user passed filter name are valid
                for(auto const& filterName : filter)
                {
                    if(std::find(allowedFilters.begin(), allowedFilters.end(), filterName) == allowedFilters.end())
                    {
                        throw std::runtime_error(name + ": unknown filter '" + filterName + "'");
                    }
                }
            }

            size_t getNumPlugins() const override
            {
                return notifyPeriod.size();
            }

            std::string getDescription() const override
            {
                return description;
            }

            std::string getOptionPrefix() const
            {
                return prefix;
            }

            std::string getName() const override
            {
                return name;
            }

            std::string const name = "MacroParticleCounter";
            //! short description of the plugin
            std::string const description = "create openPMD file with macro particle count per superCell";
            //! prefix used for command line arguments
            std::string const prefix = Species::FrameType::getName() + std::string("_macroParticlesPerSuperCell");
        };


    private:
        MappingDesc* m_cellDescription = nullptr;

        std::shared_ptr<Help> m_help;
        size_t m_id;
        std::string foldername;
        using SuperCellSize = MappingDesc::SuperCellSize;
        using GridBufferType = GridBuffer<size_t, simDim>;

        mpi::MPIReduce reduce{};

        std::unique_ptr<GridBufferType> localResult;

        std::optional<::openPMD::Series> m_Series;

        ::openPMD::Offset m_offset;
        ::openPMD::Extent m_extent;

    public:
        //! must be implemented by the user
        static std::shared_ptr<plugins::multi::IHelp> getHelp()
        {
            return std::shared_ptr<plugins::multi::IHelp>(new Help{});
        }

        MacroParticleCounter(
            std::shared_ptr<plugins::multi::IHelp>& help,
            size_t const id,
            MappingDesc* cellDescription)
            : m_cellDescription(cellDescription)
            , m_help(std::static_pointer_cast<Help>(help))
            , m_id(id)
        {
            // set how often the plugin should be executed while PIConGPU is running
            Environment<>::get().PluginConnector().setNotificationPeriod(this, m_help->notifyPeriod.get(id));


            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();

            DataSpace<simDim> localSuperCells(subGrid.getLocalDomain().size / SuperCellSize::toRT());
            localResult = std::make_unique<GridBufferType>(localSuperCells);

            /* create folder for openPMD files*/
            foldername = Species::FrameType::getName() + std::string("_") + m_help->filter.get(id)
                         + std::string("_macroParticlesPerSuperCell");
            pmacc::Filesystem::get().createDirectoryWithPermissions(foldername);
        }

        virtual ~MacroParticleCounter()
        {
            m_Series.reset();
        }

        void notify(uint32_t currentStep) override
        {
            countMacroParticles<CORE + BORDER>(currentStep);
        }

        void restart(uint32_t restartStep, std::string const& restartDirectory) override
        {
        }

        void checkpoint(uint32_t currentStep, std::string const& checkpointDirectory) override
        {
        }

        template<uint32_t AREA>
        void countMacroParticles(uint32_t currentStep)
        {
            openSeries();

            DataConnector& dc = Environment<>::get().DataConnector();

            auto particles = dc.get<Species>(Species::FrameType::getName());
            auto idProvider = dc.get<IdProvider>("globalId");

            /*############ count particles #######################################*/
            using SuperCellSize = MappingDesc::SuperCellSize;
            auto const mapper = makeAreaMapper<AREA>(*m_cellDescription);


            auto kernel = PMACC_LOCKSTEP_KERNEL(KernelCountMacroParticles{}).config(mapper.getGridDim(), *particles);

            auto bindKernel = std::bind(
                kernel,
                particles->getDeviceParticlesBox(),
                localResult->getDeviceBuffer().getDataBox(),
                mapper,
                std::placeholders::_1);

            meta::ForEach<typename Help::EligibleFilters, plugins::misc::ExecuteIfNameIsEqual<boost::mpl::_1>>{}(
                m_help->filter.get(m_id),
                currentStep,
                idProvider->getDeviceGenerator(),
                bindKernel);


            PMACC_LOCKSTEP_KERNEL(KernelCountMacroParticles{}).config(mapper.getGridDim(), *particles)();

            localResult->deviceToHost();


            /*############ dump data #############################################*/
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();

            DataSpace<simDim> localDomainSize(subGrid.getLocalDomain().size / SuperCellSize::toRT());
            DataSpace<simDim> localDomainOffset(subGrid.getLocalDomain().offset / SuperCellSize::toRT());
            DataSpace<simDim> globalDomainSize(subGrid.getGlobalDomain().size / SuperCellSize::toRT());

            ::openPMD::Extent openPmdGlobalDomainExtent(simDim);

            ::openPMD::Extent openPmdLocalDomainOffset(simDim);
            ::openPMD::Offset openPmdLocalDomainExtent(simDim);

            for(::openPMD::Extent::value_type d = 0; d < simDim; ++d)
            {
                openPmdGlobalDomainExtent[simDim - d - 1] = globalDomainSize[d];
                openPmdLocalDomainOffset[simDim - d - 1] = localDomainOffset[d];
                openPmdLocalDomainExtent[simDim - d - 1] = localDomainSize[d];
            }

            size_t* ptr = localResult->getHostBuffer().data();

            // avoid deadlock between not finished pmacc tasks and collective or blocking MPI calls in openPMD
            eventSystem::getTransactionEvent().waitForFinished();

            auto iteration = m_Series->writeIterations()[currentStep];

            auto mesh = iteration.meshes["makroParticlePerSupercell"];
            auto dataset = mesh[::openPMD::RecordComponent::SCALAR];

            openPMD::SetMeshAttributes setMeshAttributes(currentStep);
            // gridSpacing = SuperCellSize::toRT() * cellSize
            // m_gridSpacing is initialized by the cellSize
            {
                auto superCellSize = SuperCellSize::toRT();
                for(uint32_t d = 0; d < simDim; ++d)
                {
                    setMeshAttributes.m_gridSpacing[simDim - d - 1] *= superCellSize[d];
                }
            }

            setMeshAttributes(mesh)(dataset);

            dataset.resetDataset({::openPMD::determineDatatype<size_t>(), openPmdGlobalDomainExtent});
            dataset.storeChunk(
                std::shared_ptr<size_t>{ptr, [](auto const*) {}},
                openPmdLocalDomainOffset,
                openPmdLocalDomainExtent);

            openPMD::WriteMeta writeMetaAttributes;
            writeMetaAttributes(
                *m_Series,
                iteration,
                currentStep,
                /* writeFieldMeta = */ false,
                /* writeParticleMeta = */ false,
                /* writeToLog = */ false);

            iteration.close();
        }

        void openSeries()
        {
            if(!m_Series)
            {
                GridController<simDim>& gc = Environment<simDim>::get().GridController();

                std::string infix = m_help->m_filenameInfix.get(m_id);
                if(infix == "NULL")
                {
                    infix = "";
                }
                std::string filename = foldername + std::string("/macroParticlePerSupercell") + infix
                                       + std::string(".") + m_help->file_name_extension.get(m_id);
                log<picLog::INPUT_OUTPUT>("openPMD open Series at: %1%") % filename;

                m_Series = ::openPMD::Series(filename, ::openPMD::Access::CREATE, gc.getCommunicator().getMPIComm());
            }
        }
    };

} // namespace picongpu

PIC_REGISTER_SPECIES_PLUGIN(picongpu::plugins::multi::Master<picongpu::MacroParticleCounter<boost::mpl::_1>>);
#endif
