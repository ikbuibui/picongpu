/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <caravan/mpi/error.hpp>
#include <caravan/mpi/runtime.hpp>
#include <mpi.h>

namespace caravan
{
    using detail::mpiError;

    class MpiExternalRuntime::Impl
    {
    public:
        Impl()
        {
            int initialized = 0;
            int finalized = 0;
            if(MPI_Initialized(&initialized) != MPI_SUCCESS || MPI_Finalized(&finalized) != MPI_SUCCESS || !initialized
               || finalized)
                throw std::logic_error("MpiExternalRuntime requires an active caller-owned MPI lifecycle");

            int const getHandlerError = MPI_Comm_get_errhandler(MPI_COMM_WORLD, &m_previousErrorHandler);
            if(getHandlerError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_get_errhandler", getHandlerError);
            try
            {
                int const setHandlerError = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
                if(setHandlerError != MPI_SUCCESS)
                    throw mpiError("MPI_Comm_set_errhandler", setHandlerError);
                context.reset(new MpiContext{});
            }
            catch(...)
            {
                restoreErrorHandler();
                throw;
            }
        }

        ~Impl()
        {
            if(context && !context->shutdownComplete())
                std::terminate();
            context.reset();
            restoreErrorHandler();
        }

        void restoreErrorHandler() noexcept
        {
            if(m_previousErrorHandler == MPI_ERRHANDLER_NULL)
                return;
            auto handler = std::exchange(m_previousErrorHandler, MPI_ERRHANDLER_NULL);
            int const setError = MPI_Comm_set_errhandler(MPI_COMM_WORLD, handler);
            int const freeError = MPI_Errhandler_free(&handler);
            if(setError != MPI_SUCCESS || freeError != MPI_SUCCESS)
                std::terminate();
        }

        std::unique_ptr<MpiContext> context;

    private:
        MPI_Errhandler m_previousErrorHandler = MPI_ERRHANDLER_NULL;
    };

    MpiExternalRuntime::MpiExternalRuntime() : m_implementation(std::make_unique<Impl>())
    {
    }

    MpiExternalRuntime::~MpiExternalRuntime() = default;

    MpiContext& MpiExternalRuntime::context() noexcept
    {
        return *m_implementation->context;
    }

    bool MpiExternalRuntime::progress()
    {
        return m_implementation->context->progress();
    }

    void MpiExternalRuntime::requestShutdown()
    {
        m_implementation->context->requestShutdown();
    }

    void MpiExternalRuntime::finish()
    {
        requestShutdown();
        while(progress())
            std::this_thread::yield();
    }

    int MpiRuntime::runImpl(int& argc, char**& argv, std::function<int(MpiContext&)> application)
    {
        std::promise<MpiContext*> startup;
        auto ready = startup.get_future();
        std::exception_ptr workerError;
        std::jthread mpiWorker(
            [&]
            {
                bool initialized = false;
                bool published = false;
                try
                {
                    int provided = MPI_THREAD_SINGLE;
                    int const initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
                    if(initError != MPI_SUCCESS)
                        throw std::runtime_error(
                            "MPI_Init_thread failed with error code " + std::to_string(initError));
                    initialized = true;
                    if(provided < MPI_THREAD_FUNNELED)
                        throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");

                    int const handlerError = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
                    if(handlerError != MPI_SUCCESS)
                        throw mpiError("MPI_Comm_set_errhandler", handlerError);

                    MpiContext context;
                    startup.set_value(&context);
                    published = true;
                    context.run();

                    int const finalizeError = MPI_Finalize();
                    initialized = false;
                    if(finalizeError != MPI_SUCCESS)
                        throw std::runtime_error(
                            "MPI_Finalize failed with error code " + std::to_string(finalizeError));
                }
                catch(...)
                {
                    auto const error = std::current_exception();
                    if(initialized)
                        MPI_Finalize();
                    if(published)
                        workerError = error;
                    else
                        startup.set_exception(error);
                }
            });

        MpiContext* context;
        try
        {
            context = ready.get();
        }
        catch(...)
        {
            mpiWorker.join();
            throw;
        }

        int applicationResult = 0;
        std::exception_ptr applicationError;
        try
        {
            applicationResult = application(*context);
        }
        catch(...)
        {
            applicationError = std::current_exception();
        }
        context->requestShutdown();
        mpiWorker.join();

        if(applicationError)
            std::rethrow_exception(applicationError);
        if(workerError)
            std::rethrow_exception(workerError);
        return applicationResult;
    }
} // namespace caravan
