#pragma once

#include <pmacc/pluginSystem/IPlugin.hpp>

#include <cstdint>
#include <string>

namespace sph
{
    /**
     * Interface for a lightweight simulation plugin
     * without checkpoint/restart capabilities.
     */
    class ILightweightPlugin : public pmacc::IPlugin
    {
    public:
        void restart(uint32_t, std::string const) override
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        void checkpoint(uint32_t, std::string const) override
        {
            // disable checkpoint/restart capabilities for lightweight plugins
        }

        ~ILightweightPlugin() override = default;
    };
} // namespace sph
