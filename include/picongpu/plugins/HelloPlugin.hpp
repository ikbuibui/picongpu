#pragma once

#include "picongpu/plugins/ISimulationPlugin.hpp"

namespace picongpu
{
    using namespace pmacc;


    class HelloPlugin : public ISimulationPlugin
    {
    private:
        MappingDesc* cellDescription;
        std::string pluginName;

    public:
        // constructor doesn't know command line arguments if any, use pluginLoad to finish initialization
        HelloPlugin()
            : pluginName{"HelloPlugin: A simple hello world plugin!"}
            // , pluginPrefix(std::string("fields_energy"))
            // , filename(pluginPrefix + ".dat")
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        ~HelloPlugin() override = default;

        void pluginRegisterHelp(po::options_description& desc) override
        {
            return;
        }

        std::string pluginGetName() const override
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription) override
        {
            this->cellDescription = cellDescription;
        }

        void notify(uint32_t currentStep) override
        {
            printf("Hi! The current step is %d \n", currentStep);
        }

        void restart(uint32_t restartStep, const std::string restartDirectory)
        {
            /* restart from a checkpoint here
            * will be called only once per simulation and before notify() */
        }

        void checkpoint(uint32_t currentStep, const std::string restartDirectory)
        {
            /* create a persistent checkpoint here
            * will be called before notify() if both will be called for the same timestep */
        }

    private:
        void pluginLoad() override
        {
            Environment<>::get().PluginConnector().setNotificationPeriod(this, "2");
        }
    };
}