/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <cassert>
#include <climits>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
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
            int const rankError = MPI_Comm_rank(MPI_COMM_WORLD, &m_topology.rank);
            if(rankError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_rank", rankError);
            int const sizeError = MPI_Comm_size(MPI_COMM_WORLD, &m_topology.size);
            if(sizeError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_size", sizeError);
        }

        TopologySnapshot topology() const
        {
            return m_topology;
        }

        Future<SendResult> send(
            Event predecessor,
            BufferLease buffer,
            Peer destination,
            MessageTag tag,
            CommunicatorId communicator)
        {
            Promise<SendResult> completion;
            auto result = completion.future();
            if(!validBuffer(buffer) || destination.any || destination.value < 0 || tag.any || tag.value < 0
               || communicator != worldCommunicator)
            {
                completion.setFailed(std::make_exception_ptr(std::invalid_argument("Invalid Caravan MPI send")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, buffer = std::move(buffer), destination, tag](Promise<SendResult> output) mutable
                { startSend(std::move(output), std::move(buffer), destination, tag); });
            return result;
        }

        Future<ReceiveResult> receive(
            Event predecessor,
            BufferLease buffer,
            Peer source,
            MessageTag tag,
            CommunicatorId communicator)
        {
            Promise<ReceiveResult> completion;
            auto result = completion.future();
            if(!validBuffer(buffer) || (!source.any && source.value < 0) || (!tag.any && tag.value < 0)
               || communicator != worldCommunicator)
            {
                completion.setFailed(std::make_exception_ptr(std::invalid_argument("Invalid Caravan MPI receive")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, buffer = std::move(buffer), source, tag](Promise<ReceiveResult> output) mutable
                { startReceive(std::move(output), std::move(buffer), source, tag); });
            return result;
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
            submitAfter(
                std::move(predecessor),
                completion,
                [this](EventSource output) { startBarrier(std::move(output)); });
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

        struct BarrierCompletion
        {
            EventSource output;
        };

        struct SendCompletion
        {
            Promise<SendResult> output;
            std::size_t bytes;
        };

        struct ReceiveCompletion
        {
            Promise<ReceiveResult> output;
        };

        using ActiveCompletion = std::variant<BarrierCompletion, SendCompletion, ReceiveCompletion>;

        struct ActiveOperation
        {
            ActiveCompletion completion;
            std::optional<BufferLease> buffer;
        };

        static bool validBuffer(BufferLease const& buffer)
        {
            return buffer.valid() && buffer.bytes() <= static_cast<std::size_t>(INT_MAX);
        }

        template<typename T_Output, typename T_Start>
        void submitAfter(Event predecessor, T_Output output, T_Start&& start)
        {
            {
                std::lock_guard lock(m_queueMutex);
                if(!m_accepting)
                {
                    output.setFailed(std::make_exception_ptr(std::runtime_error("MPI executor is shutting down")));
                    return;
                }
                ++m_outstanding;
            }

            predecessor.continueWith(
                m_continuations,
                [this, predecessor, output, start = std::forward<T_Start>(start)](Event) mutable
                {
                    if(predecessor.state() == CompletionState::failed)
                    {
                        output.setFailed(predecessor.error());
                        finishOperation();
                    }
                    else if(predecessor.state() == CompletionState::cancelled)
                    {
                        output.cancel();
                        finishOperation();
                    }
                    else
                    {
                        try
                        {
                            std::invoke(start, output);
                        }
                        catch(...)
                        {
                            output.setFailed(std::current_exception());
                            finishOperation();
                        }
                    }
                });
        }

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

        void prepareActive()
        {
            auto const capacity = m_requests.size() + 1u;
            m_requests.reserve(capacity);
            m_active.reserve(capacity);
        }

        void startBarrier(EventSource completion)
        {
            assertOwner();
            prepareActive();
            MPI_Request request = MPI_REQUEST_NULL;
            int const error = MPI_Ibarrier(MPI_COMM_WORLD, &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Ibarrier", error)));
                finishOperation();
                return;
            }
            m_requests.emplace_back(request);
            m_active.emplace_back(BarrierCompletion{std::move(completion)}, std::nullopt);
        }

        void startSend(Promise<SendResult> completion, BufferLease buffer, Peer destination, MessageTag tag)
        {
            assertOwner();
            prepareActive();
            MPI_Request request = MPI_REQUEST_NULL;
            int const error = MPI_Isend(
                buffer.data(),
                static_cast<int>(buffer.bytes()),
                MPI_BYTE,
                destination.value,
                tag.value,
                MPI_COMM_WORLD,
                &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Isend", error)));
                finishOperation();
                return;
            }
            auto const bytes = buffer.bytes();
            m_requests.emplace_back(request);
            m_active.emplace_back(SendCompletion{std::move(completion), bytes}, std::move(buffer));
        }

        void startReceive(Promise<ReceiveResult> completion, BufferLease buffer, Peer source, MessageTag tag)
        {
            assertOwner();
            prepareActive();
            MPI_Request request = MPI_REQUEST_NULL;
            int const error = MPI_Irecv(
                buffer.data(),
                static_cast<int>(buffer.bytes()),
                MPI_BYTE,
                source.any ? MPI_ANY_SOURCE : source.value,
                tag.any ? MPI_ANY_TAG : tag.value,
                MPI_COMM_WORLD,
                &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Irecv", error)));
                finishOperation();
                return;
            }
            m_requests.emplace_back(request);
            m_active.emplace_back(ReceiveCompletion{std::move(completion)}, std::move(buffer));
        }

        void failActive(ActiveOperation& active, std::exception_ptr failure)
        {
            std::visit([&](auto& completion) { completion.output.setFailed(failure); }, active.completion);
            finishOperation();
        }

        void completeActive(ActiveOperation& active, MPI_Status const& status)
        {
            std::visit(
                [&](auto& completion)
                {
                    using Completion = std::remove_cvref_t<decltype(completion)>;
                    if constexpr(std::is_same_v<Completion, BarrierCompletion>)
                        completion.output.setReady();
                    else if constexpr(std::is_same_v<Completion, SendCompletion>)
                        completion.output.setValue(SendResult{completion.bytes});
                    else
                    {
                        int bytes = 0;
                        int const error = MPI_Get_count(&status, MPI_BYTE, &bytes);
                        if(error != MPI_SUCCESS)
                            completion.output.setFailed(std::make_exception_ptr(mpiError("MPI_Get_count", error)));
                        else if(bytes == MPI_UNDEFINED)
                            completion.output.setFailed(
                                std::make_exception_ptr(std::runtime_error("MPI_Get_count returned MPI_UNDEFINED")));
                        else
                            completion.output.setValue(
                                ReceiveResult{
                                    Peer{status.MPI_SOURCE},
                                    MessageTag{status.MPI_TAG},
                                    static_cast<std::size_t>(bytes)});
                    }
                },
                active.completion);
            finishOperation();
        }

        void progress()
        {
            assertOwner();
            if(m_requests.empty())
                return;

            m_completedIndices.resize(m_requests.size());
            m_statuses.resize(m_requests.size());
            int completed = 0;
            int const error = MPI_Testsome(
                static_cast<int>(m_requests.size()),
                m_requests.data(),
                &completed,
                m_completedIndices.data(),
                m_statuses.data());
            if(error != MPI_SUCCESS)
            {
                auto failure = std::make_exception_ptr(mpiError("MPI_Testsome", error));
                for(auto& active : m_active)
                    failActive(active, failure);
                m_requests.clear();
                m_active.clear();
                return;
            }
            if(completed == MPI_UNDEFINED || completed == 0)
                return;

            for(int i = 0; i < completed; ++i)
            {
                auto const position = static_cast<std::size_t>(i);
                auto const index = static_cast<std::size_t>(m_completedIndices[position]);
                completeActive(m_active[index], m_statuses[position]);
            }

            std::size_t output = 0u;
            for(std::size_t input = 0u; input < m_requests.size(); ++input)
            {
                if(m_requests[input] == MPI_REQUEST_NULL)
                    continue;
                if(output != input)
                {
                    m_requests[output] = m_requests[input];
                    m_active[output] = std::move(m_active[input]);
                }
                ++output;
            }
            m_requests.resize(output);
            m_active.resize(output);
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
        TopologySnapshot m_topology{};
        std::mutex m_queueMutex;
        std::condition_variable m_queueReady;
        std::deque<std::function<void()>> m_queue;
        std::size_t m_outstanding = 0u;
        bool m_accepting = true;
        bool m_stopping = false;
        std::vector<MPI_Request> m_requests;
        std::vector<ActiveOperation> m_active;
        std::vector<int> m_completedIndices;
        std::vector<MPI_Status> m_statuses;
        ContinuationTarget m_continuations;
    };

    MpiExecutor::MpiExecutor(std::unique_ptr<Impl> implementation) : m_implementation(std::move(implementation))
    {
    }

    MpiExecutor::~MpiExecutor() = default;

    TopologySnapshot MpiExecutor::topology() const
    {
        return m_implementation->topology();
    }

    Future<SendResult> MpiExecutor::send(
        Event dataReady,
        BufferLease buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        return m_implementation->send(std::move(dataReady), std::move(buffer), destination, tag, communicator);
    }

    Future<ReceiveResult> MpiExecutor::receive(
        Event bufferAvailable,
        BufferLease buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        return m_implementation->receive(std::move(bufferAvailable), std::move(buffer), source, tag, communicator);
    }

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

        std::unique_ptr<MpiExecutor::Impl> implementation;
        try
        {
            implementation = std::make_unique<MpiExecutor::Impl>();
        }
        catch(...)
        {
            MPI_Finalize();
            throw;
        }
        MpiExecutor executor{std::move(implementation)};
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
