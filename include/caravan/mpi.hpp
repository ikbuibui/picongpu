/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <caravan/core.hpp>

namespace caravan
{
    struct CommunicatorId
    {
        std::uint32_t value;

        friend bool operator==(CommunicatorId const&, CommunicatorId const&) = default;
    };

    inline constexpr CommunicatorId worldCommunicator{0u};

    class MpiExecutor
    {
    public:
        MpiExecutor(MpiExecutor const&) = delete;
        MpiExecutor& operator=(MpiExecutor const&) = delete;
        ~MpiExecutor();

        Event barrier(Event predecessor, CommunicatorId communicator = worldCommunicator);

    private:
        class Impl;
        explicit MpiExecutor(std::unique_ptr<Impl> implementation);

        void run();
        void requestShutdown();

        std::unique_ptr<Impl> m_implementation;

        friend class MpiRuntime;
    };

    class MpiRuntime
    {
    public:
        template<typename T_Application>
        static int run(int& argc, char**& argv, T_Application&& application)
        {
            auto invoke = [&application](MpiExecutor& executor)
            {
                if constexpr(std::is_invocable_v<T_Application&, MpiExecutor&>)
                {
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&, MpiExecutor&>>)
                    {
                        std::invoke(application, executor);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application, executor));
                }
                else
                {
                    static_assert(std::is_invocable_v<T_Application&>);
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&>>)
                    {
                        std::invoke(application);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application));
                }
            };
            return runImpl(argc, argv, invoke);
        }

    private:
        static int runImpl(int& argc, char**& argv, std::function<int(MpiExecutor&)> application);
    };
} // namespace caravan
