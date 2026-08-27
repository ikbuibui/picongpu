/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <cassert>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <caravan/mpi.hpp>
#include <mpi.h>

namespace caravan
{
    namespace
    {
        std::runtime_error mpiError(char const* operation, int errorCode)
        {
            char message[MPI_MAX_ERROR_STRING];
            int length = 0;
            MPI_Error_string(errorCode, message, &length);
            return std::runtime_error(
                std::string{operation} + ": " + std::string{message, static_cast<std::size_t>(length)});
        }
    } // namespace

    class MpiExecutor::Impl
    {
    public:
        Impl() : m_owner(std::this_thread::get_id()), m_continuations(*this)
        {
        }

        Event barrier(Event predecessor, CommunicatorId communicator)
        {
            EventSource completion;
            auto result = completion.event();
            if(communicator != worldCommunicator)
            {
                completion.setFailed(std::make_exception_ptr(std::invalid_argument("Unknown Caravan communicator")));
                return result;
            }

            {
                std::lock_guard lock(m_queueMutex);
                if(!m_accepting)
                {
                    completion.setFailed(std::make_exception_ptr(std::runtime_error("MPI executor is shutting down")));
                    return result;
                }
                ++m_outstanding;
            }

            predecessor.continueWith(
                m_continuations,
                [this, predecessor, completion](Event)
                {
                    if(predecessor.state() == CompletionState::failed)
                    {
                        completion.setFailed(predecessor.error());
                        finishOperation();
                    }
                    else if(predecessor.state() == CompletionState::cancelled)
                    {
                        completion.cancel();
                        finishOperation();
                    }
                    else
                        startBarrier(completion);
                });
            return result;
        }

        void run()
        {
            assertOwner();
            ExecutorThreadGuard guard;
            for(;;)
            {
                drainQueue();
                progress();

                std::unique_lock lock(m_queueMutex);
                if(m_stopping && m_outstanding == 0u)
                    return;
                if(m_requests.empty() && m_queue.empty())
                    m_queueReady.wait(
                        lock,
                        [this] { return !m_queue.empty() || (m_stopping && m_outstanding == 0u); });
            }
        }

        void requestShutdown()
        {
            {
                std::lock_guard lock(m_queueMutex);
                m_accepting = false;
                m_stopping = true;
            }
            m_queueReady.notify_one();
        }

    private:
        class ContinuationTarget
        {
        public:
            explicit ContinuationTarget(Impl& owner) : m_owner(owner)
            {
            }

            void post(std::function<void()> continuation)
            {
                m_owner.enqueue(std::move(continuation));
            }

        private:
            Impl& m_owner;
        };

        void assertOwner() const
        {
            assert(std::this_thread::get_id() == m_owner && "MPI operation executed outside the MPI owner thread");
        }

        void enqueue(std::function<void()> command)
        {
            {
                std::lock_guard lock(m_queueMutex);
                m_queue.emplace_back(std::move(command));
            }
            m_queueReady.notify_one();
        }

        void drainQueue()
        {
            assertOwner();
            std::deque<std::function<void()>> commands;
            {
                std::lock_guard lock(m_queueMutex);
                commands.swap(m_queue);
            }
            for(auto& command : commands)
                command();
        }

        void startBarrier(EventSource completion)
        {
            assertOwner();
            MPI_Request request = MPI_REQUEST_NULL;
            int const error = MPI_Ibarrier(MPI_COMM_WORLD, &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Ibarrier", error)));
                finishOperation();
                return;
            }
            m_requests.emplace_back(request);
            m_completions.emplace_back(std::move(completion));
        }

        void progress()
        {
            assertOwner();
            if(m_requests.empty())
                return;

            std::vector<int> indices(m_requests.size());
            std::vector<MPI_Status> statuses(m_requests.size());
            int completed = 0;
            int const error = MPI_Testsome(
                static_cast<int>(m_requests.size()),
                m_requests.data(),
                &completed,
                indices.data(),
                statuses.data());
            if(error != MPI_SUCCESS)
            {
                auto failure = std::make_exception_ptr(mpiError("MPI_Testsome", error));
                for(auto const& completion : m_completions)
                {
                    completion.setFailed(failure);
                    finishOperation();
                }
                m_requests.clear();
                m_completions.clear();
                return;
            }
            if(completed == MPI_UNDEFINED || completed == 0)
                return;

            for(int i = 0; i < completed; ++i)
            {
                auto const index = static_cast<std::size_t>(indices[static_cast<std::size_t>(i)]);
                m_completions[index].setReady();
                finishOperation();
            }

            std::size_t output = 0u;
            for(std::size_t input = 0u; input < m_requests.size(); ++input)
            {
                if(m_requests[input] == MPI_REQUEST_NULL)
                    continue;
                if(output != input)
                {
                    m_requests[output] = m_requests[input];
                    m_completions[output] = std::move(m_completions[input]);
                }
                ++output;
            }
            m_requests.resize(output);
            m_completions.resize(output);
        }

        void finishOperation()
        {
            {
                std::lock_guard lock(m_queueMutex);
                --m_outstanding;
            }
            m_queueReady.notify_one();
        }

        std::thread::id m_owner;
        std::mutex m_queueMutex;
        std::condition_variable m_queueReady;
        std::deque<std::function<void()>> m_queue;
        std::size_t m_outstanding = 0u;
        bool m_accepting = true;
        bool m_stopping = false;
        std::vector<MPI_Request> m_requests;
        std::vector<EventSource> m_completions;
        ContinuationTarget m_continuations;
    };

    MpiExecutor::MpiExecutor(std::unique_ptr<Impl> implementation) : m_implementation(std::move(implementation))
    {
    }

    MpiExecutor::~MpiExecutor() = default;

    Event MpiExecutor::barrier(Event predecessor, CommunicatorId communicator)
    {
        return m_implementation->barrier(std::move(predecessor), communicator);
    }

    void MpiExecutor::run()
    {
        m_implementation->run();
    }

    void MpiExecutor::requestShutdown()
    {
        m_implementation->requestShutdown();
    }

    int MpiRuntime::runImpl(int& argc, char**& argv, std::function<int(MpiExecutor&)> application)
    {
        int provided = MPI_THREAD_SINGLE;
        int const initError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if(initError != MPI_SUCCESS)
            throw std::runtime_error("MPI_Init_thread failed with error code " + std::to_string(initError));
        if(provided < MPI_THREAD_FUNNELED)
        {
            MPI_Finalize();
            throw std::runtime_error("MPI does not provide MPI_THREAD_FUNNELED");
        }

        int const handlerError = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        if(handlerError != MPI_SUCCESS)
        {
            auto error = mpiError("MPI_Comm_set_errhandler", handlerError);
            MPI_Finalize();
            throw error;
        }

        MpiExecutor executor{std::make_unique<MpiExecutor::Impl>()};
        int applicationResult = 0;
        std::exception_ptr applicationError;
        std::thread applicationThread;
        try
        {
            applicationThread = std::thread(
                [&]
                {
                    try
                    {
                        applicationResult = application(executor);
                    }
                    catch(...)
                    {
                        applicationError = std::current_exception();
                    }
                    executor.requestShutdown();
                });
        }
        catch(...)
        {
            MPI_Finalize();
            throw;
        }

        executor.run();
        applicationThread.join();
        int const finalizeError = MPI_Finalize();
        if(applicationError)
            std::rethrow_exception(applicationError);
        if(finalizeError != MPI_SUCCESS)
            throw std::runtime_error("MPI_Finalize failed with error code " + std::to_string(finalizeError));
        return applicationResult;
    }
} // namespace caravan
