/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#pragma once

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <caravan/mpi/context.hpp>

namespace caravan
{
    /** Attach Caravan's request engine to an MPI lifecycle owned by the caller.
     *
     * MPI must already be initialized. Construct and drive this object only from
     * the thread on which native MPI calls are permitted. progress() performs one
     * bounded, nonblocking submission/progress turn and returns false after a
     * requested shutdown becomes quiescent. The caller retains responsibility
     * for MPI_Finalize and must finish Caravan first.
     */
    class MpiExternalRuntime
    {
    public:
        MpiExternalRuntime();
        ~MpiExternalRuntime();

        MpiExternalRuntime(MpiExternalRuntime const&) = delete;
        MpiExternalRuntime& operator=(MpiExternalRuntime const&) = delete;

        MpiContext& context() noexcept;
        bool progress();
        void requestShutdown();
        void finish();

    private:
        class Impl;
        std::unique_ptr<Impl> m_implementation;
    };

    /** Convenience MPI lifecycle/driver using a dedicated FUNNELED thread. */
    class MpiRuntime
    {
    public:
        template<typename T_Application>
        static int run(int& argc, char**& argv, T_Application&& application)
        {
            auto invoke = [&application](MpiContext& context)
            {
                if constexpr(std::is_invocable_v<T_Application&, MpiContext&>)
                {
                    if constexpr(std::is_void_v<std::invoke_result_t<T_Application&, MpiContext&>>)
                    {
                        std::invoke(application, context);
                        return 0;
                    }
                    else
                        return static_cast<int>(std::invoke(application, context));
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
        static int runImpl(int& argc, char**& argv, std::function<int(MpiContext&)> application);
    };
} // namespace caravan
