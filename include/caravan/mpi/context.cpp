/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <caravan/core/eager.hpp>
#include <caravan/mpi/error.hpp>
#include <caravan/mpi/native.hpp>
#include <mpi.h>

namespace caravan
{
    using detail::mpiError;

    class MpiContext::Impl
    {
    public:
        Impl() : m_owner(std::this_thread::get_id())
        {
            m_topology.communicator = worldCommunicator;
            int const rankError = MPI_Comm_rank(MPI_COMM_WORLD, &m_topology.rank);
            if(rankError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_rank", rankError);
            int const sizeError = MPI_Comm_size(MPI_COMM_WORLD, &m_topology.size);
            if(sizeError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_size", sizeError);

            MPI_Comm host = MPI_COMM_NULL;
            int const splitError
                = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, m_topology.rank, MPI_INFO_NULL, &host);
            if(splitError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_split_type", splitError);
            int const hostRankError = MPI_Comm_rank(host, &m_topology.hostLocalRank);
            int const freeError = MPI_Comm_free(&host);
            if(hostRankError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_rank(host)", hostRankError);
            if(freeError != MPI_SUCCESS)
                throw mpiError("MPI_Comm_free(host)", freeError);
        }

        TopologySnapshot topology() const
        {
            return m_topology;
        }

        void submitNative(detail::NativeSubmission submission)
        {
            if(detail::nativeCallbackDepth != 0u)
            {
                submission.setFailed(
                    std::make_exception_ptr(std::logic_error("Recursive native MPI submission is not allowed")));
                return;
            }
            submit(std::move(submission), [this](detail::NativeSubmission output) { startNative(std::move(output)); });
        }

        void invokeBlocking(detail::NativeBlockingSubmission submission)
        {
            if(detail::nativeCallbackDepth != 0u)
            {
                submission.setFailed(
                    std::make_exception_ptr(std::logic_error("Recursive native MPI submission is not allowed")));
                return;
            }
            submit(
                std::move(submission),
                [this](detail::NativeBlockingSubmission output) { startBlocking(std::move(output)); });
        }

        detail::ManagedCollectiveTicket reserveManagedCollective(CommunicatorId communicator)
        {
            std::lock_guard lock(m_queueMutex);
            if(!m_accepting)
                throw std::runtime_error("MPI context is shutting down");
            auto& lane = m_managedCollectives[communicator.value];
            auto const ticket = lane.reserve(communicator);
            ++m_outstanding;
            return ticket;
        }

        void releaseManagedCollective(detail::ManagedCollectiveTicket ticket, std::function<void()> start)
        {
            {
                std::lock_guard lock(m_queueMutex);
                auto lane = m_managedCollectives.find(ticket.communicator.value);
                if(lane == m_managedCollectives.end())
                    throw std::logic_error("Unknown managed collective ticket");
                lane->second.commit(ticket.sequence, std::move(start));
            }
            m_queueReady.notify_one();
        }

        void abandonManagedCollective(detail::ManagedCollectiveTicket ticket) noexcept
        {
            {
                std::lock_guard lock(m_queueMutex);
                auto lane = m_managedCollectives.find(ticket.communicator.value);
                if(lane == m_managedCollectives.end())
                    return;
                lane->second.skip(ticket.sequence);
            }
            m_queueReady.notify_one();
        }

        void run()
        {
            assertOwner();
            ExecutorThreadGuard guard;
            for(;;)
            {
                drainQueue();
                drainManagedCollectives();
                progress();

                std::unique_lock lock(m_queueMutex);
                if(m_stopping && m_outstanding == 0u)
                {
                    lock.unlock();
                    releaseCommunicators();
                    return;
                }
                if(m_requests.empty() && m_queue.empty() && !hasReadyManagedCollective())
                    m_queueReady.wait(
                        lock,
                        [this]
                        {
                            return !m_queue.empty() || hasReadyManagedCollective()
                                   || (m_stopping && m_outstanding == 0u);
                        });
            }
        }

        void requestShutdown()
        {
            {
                std::lock_guard lock(m_queueMutex);
                m_accepting = false;
                m_stopping = true;
                for(auto& [communicator, lane] : m_managedCollectives)
                    lane.skipReserved();
            }
            m_queueReady.notify_one();
        }

    private:
        struct NativeGroup
        {
            detail::NativeSubmission output;
            std::vector<MPI_Status> statuses;
            std::vector<std::shared_ptr<void>> lifetimes;
            std::size_t remaining;
            std::exception_ptr failure;
            bool terminal = false;

            bool retire(
                NativeMpiContext& context,
                std::size_t index,
                MPI_Status const& status,
                std::exception_ptr error = {})
            {
                if(terminal)
                    return false;
                statuses[index] = status;
                if(error && !failure)
                    failure = std::move(error);
                if(--remaining != 0u)
                    return false;
                terminal = true;
                if(failure)
                    output.failed(std::move(failure));
                else
                {
                    try
                    {
                        output.completed(context, statuses);
                    }
                    catch(...)
                    {
                        output.failed(std::current_exception());
                    }
                }
                return true;
            }
        };

        struct NativeCompletion
        {
            std::shared_ptr<NativeGroup> group;
            std::size_t index;
        };

        struct ManagedCollectiveLane
        {
            enum class State : std::uint8_t
            {
                reserved,
                committed,
                skipped
            };

            struct Entry
            {
                std::function<void()> start;
                State state = State::reserved;
            };

            /* The deque contains every non-retired ticket in contiguous sequence
             * order. Only the front may retire, and only after commit or skip.
             * Each entry contributes one to m_outstanding until popReady. These
             * transitions and all accesses happen under m_queueMutex; callbacks
             * run only after the lock is released.
             */
            detail::ManagedCollectiveTicket reserve(CommunicatorId communicator)
            {
                auto const sequence = firstSequence + entries.size();
                entries.emplace_back();
                return {communicator, sequence};
            }

            void commit(std::size_t sequence, std::function<void()> start)
            {
                auto* entry = find(sequence);
                if(entry == nullptr || entry->state != State::reserved)
                    throw std::logic_error("Inactive managed collective ticket");
                entry->start = std::move(start);
                entry->state = State::committed;
            }

            void skip(std::size_t sequence) noexcept
            {
                auto* entry = find(sequence);
                if(entry != nullptr && entry->state == State::reserved)
                    entry->state = State::skipped;
            }

            void skipReserved() noexcept
            {
                for(auto& entry : entries)
                    if(entry.state == State::reserved)
                        entry.state = State::skipped;
            }

            bool ready() const noexcept
            {
                return !entries.empty() && entries.front().state != State::reserved;
            }

            std::optional<std::function<void()>> popReady()
            {
                if(!ready())
                    return std::nullopt;
                std::optional<std::function<void()>> start{std::move(entries.front().start)};
                entries.pop_front();
                ++firstSequence;
                return start;
            }

        private:
            Entry* find(std::size_t sequence) noexcept
            {
                if(sequence < firstSequence || sequence - firstSequence >= entries.size())
                    return nullptr;
                return &entries[sequence - firstSequence];
            }

            std::size_t firstSequence = 0u;
            std::deque<Entry> entries;
        };

        template<typename T_Output, typename T_Start>
        void submit(T_Output output, T_Start&& start)
        {
            auto fail = output.failed;
            std::function<void()> command;
            try
            {
                command = [this, output = std::move(output), start = std::forward<T_Start>(start)]() mutable
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
                };
            }
            catch(...)
            {
                fail(std::current_exception());
                return;
            }

            bool accepted = false;
            try
            {
                std::lock_guard lock(m_queueMutex);
                if(m_accepting)
                {
                    m_queue.emplace_back(std::move(command));
                    ++m_outstanding;
                    accepted = true;
                }
            }
            catch(...)
            {
                fail(std::current_exception());
                return;
            }
            if(!accepted)
            {
                fail(std::make_exception_ptr(std::runtime_error("MPI context is shutting down")));
                return;
            }
            m_queueReady.notify_one();
        }

        bool hasReadyManagedCollective() const
        {
            for(auto const& [communicator, lane] : m_managedCollectives)
                if(lane.ready())
                    return true;
            return false;
        }

        void drainManagedCollectives()
        {
            assertOwner();
            for(;;)
            {
                std::optional<std::function<void()>> start;
                {
                    std::lock_guard lock(m_queueMutex);
                    for(auto& [communicator, lane] : m_managedCollectives)
                    {
                        start = lane.popReady();
                        if(start)
                        {
                            --m_outstanding;
                            break;
                        }
                    }
                }
                if(!start)
                    return;
                if(*start)
                    (*start)();
            }
        }

        void assertOwner() const
        {
            assert(std::this_thread::get_id() == m_owner && "MPI operation executed outside the MPI owner thread");
        }

        void drainQueue()
        {
            assertOwner();
            for(;;)
            {
                std::function<void()> command;
                {
                    std::lock_guard lock(m_queueMutex);
                    if(m_queue.empty())
                        return;
                    command = std::move(m_queue.front());
                    m_queue.pop_front();
                }
                command();
            }
        }

        MPI_Comm communicator(CommunicatorId id) const
        {
            if(id.value >= m_communicators.size() || m_communicators[id.value] == MPI_COMM_NULL)
                throw std::invalid_argument("Unknown Caravan communicator");
            return m_communicators[id.value];
        }

        CommunicatorId adoptCommunicator(MPI_Comm native)
        {
            assertOwner();
            if(native == MPI_COMM_NULL)
                throw std::invalid_argument("Cannot adopt MPI_COMM_NULL");
            if(m_communicators.size() >= std::numeric_limits<std::uint32_t>::max())
                throw std::overflow_error("Too many Caravan communicators");
            m_communicators.reserve(m_communicators.size() + 1u);
            int const error = MPI_Comm_set_errhandler(native, MPI_ERRORS_RETURN);
            if(error != MPI_SUCCESS)
            {
                MPI_Comm_free(&native);
                throw mpiError("MPI_Comm_set_errhandler", error);
            }
            auto const id = CommunicatorId{static_cast<std::uint32_t>(m_communicators.size())};
            m_communicators.emplace_back(native);
            return id;
        }

        void destroyCommunicator(CommunicatorId id)
        {
            assertOwner();
            if(id == worldCommunicator || id.value >= m_communicators.size()
               || m_communicators[id.value] == MPI_COMM_NULL)
                throw std::invalid_argument("Unknown or immutable Caravan communicator");
            int const error = MPI_Comm_free(&m_communicators[id.value]);
            if(error != MPI_SUCCESS)
                throw mpiError("MPI_Comm_free", error);
        }

        NativeMpiContext nativeContext()
        {
            return detail::NativeContextFactory::create(
                this,
                [](void* implementation, CommunicatorId id)
                { return static_cast<Impl*>(implementation)->communicator(id); },
                [](void* implementation, MPI_Comm native)
                { return static_cast<Impl*>(implementation)->adoptCommunicator(native); },
                [](void* implementation, CommunicatorId id)
                { static_cast<Impl*>(implementation)->destroyCommunicator(id); });
        }

        void startNative(detail::NativeSubmission output)
        {
            assertOwner();
            auto context = nativeContext();
            auto batch = output.start(context);
            output.start = {};
            auto const activeRequests = static_cast<std::size_t>(std::count_if(
                batch.requests.begin(),
                batch.requests.end(),
                [](MPI_Request request) { return request != MPI_REQUEST_NULL; }));
            m_requests.reserve(m_requests.size() + activeRequests);
            m_active.reserve(m_active.size() + activeRequests);

            if(batch.requests.empty())
            {
                detail::NativeAccess::release(batch);
                output.completed(context, {});
                finishOperation();
                return;
            }

            auto group = std::make_shared<NativeGroup>(
                output,
                std::vector<MPI_Status>(batch.requests.size()),
                std::move(batch.lifetimes),
                batch.requests.size());
            for(std::size_t index = 0u; index < batch.requests.size(); ++index)
            {
                if(batch.requests[index] == MPI_REQUEST_NULL)
                {
                    if(group->retire(context, index, MPI_Status{}))
                        finishOperation();
                    continue;
                }
                m_requests.emplace_back(batch.requests[index]);
                m_active.emplace_back(group, index);
            }
            detail::NativeAccess::release(batch);
        }

        void startBlocking(detail::NativeBlockingSubmission output)
        {
            assertOwner();
            try
            {
                auto context = nativeContext();
                output.invoke(context);
            }
            catch(...)
            {
                output.failed(std::current_exception());
            }
            finishOperation();
        }

        void releaseCommunicators()
        {
            assertOwner();
            for(std::size_t i = 1u; i < m_communicators.size(); ++i)
            {
                if(m_communicators[i] == MPI_COMM_NULL)
                    continue;
                int const error = MPI_Comm_free(&m_communicators[i]);
                if(error != MPI_SUCCESS)
                {
                    MPI_Abort(MPI_COMM_WORLD, error);
                    std::terminate();
                }
            }
        }

        void retireActive(NativeCompletion& active, MPI_Status const& status, std::exception_ptr failure = {})
        {
            auto context = nativeContext();
            if(active.group->retire(context, active.index, status, std::move(failure)))
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
            if(error != MPI_SUCCESS && error != MPI_ERR_IN_STATUS)
            {
                MPI_Abort(MPI_COMM_WORLD, error);
                std::terminate();
            }
            if(completed == MPI_UNDEFINED || completed == 0)
                return;

            for(int i = 0; i < completed; ++i)
            {
                auto const position = static_cast<std::size_t>(i);
                auto const index = static_cast<std::size_t>(m_completedIndices[position]);
                auto const requestError = error == MPI_ERR_IN_STATUS ? m_statuses[position].MPI_ERROR : MPI_SUCCESS;
                if(requestError == MPI_ERR_PENDING || m_requests[index] != MPI_REQUEST_NULL)
                    continue;
                retireActive(
                    m_active[index],
                    m_statuses[position],
                    requestError == MPI_SUCCESS ? std::exception_ptr{}
                                                : std::make_exception_ptr(mpiError("MPI request", requestError)));
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
        std::unordered_map<std::uint32_t, ManagedCollectiveLane> m_managedCollectives;
        std::vector<MPI_Comm> m_communicators{MPI_COMM_WORLD};
        std::vector<MPI_Request> m_requests;
        std::vector<NativeCompletion> m_active;
        std::vector<int> m_completedIndices;
        std::vector<MPI_Status> m_statuses;
    };

    MpiContext::MpiContext(std::unique_ptr<Impl> implementation) : m_implementation(std::move(implementation))
    {
    }

    MpiContext::~MpiContext() = default;

    TopologySnapshot MpiContext::topology() const
    {
        return m_implementation->topology();
    }

    void MpiContext::run()
    {
        m_implementation->run();
    }

    void MpiContext::requestShutdown()
    {
        m_implementation->requestShutdown();
    }

    void MpiContext::submitNative(detail::NativeSubmission submission)
    {
        m_implementation->submitNative(std::move(submission));
    }

    void MpiContext::invokeBlocking(detail::NativeBlockingSubmission submission)
    {
        m_implementation->invokeBlocking(std::move(submission));
    }

    detail::ManagedCollectiveTicket MpiContext::reserveManagedCollective(CommunicatorId communicator)
    {
        return m_implementation->reserveManagedCollective(communicator);
    }

    void MpiContext::releaseManagedCollective(detail::ManagedCollectiveTicket ticket, std::function<void()> start)
    {
        m_implementation->releaseManagedCollective(ticket, std::move(start));
    }

    void MpiContext::abandonManagedCollective(detail::ManagedCollectiveTicket ticket) noexcept
    {
        m_implementation->abandonManagedCollective(ticket);
    }

    detail::ManagedCollectiveTicket detail::CollectiveAccess::reserve(MpiContext& context, CommunicatorId communicator)
    {
        return context.reserveManagedCollective(communicator);
    }

    void detail::CollectiveAccess::release(
        MpiContext& context,
        detail::ManagedCollectiveTicket ticket,
        std::function<void()> start)
    {
        context.releaseManagedCollective(ticket, std::move(start));
    }

    void detail::CollectiveAccess::abandon(MpiContext& context, detail::ManagedCollectiveTicket ticket) noexcept
    {
        context.abandonManagedCollective(ticket);
    }

    void detail::NativeAccess::submit(MpiContext& context, detail::NativeSubmission submission)
    {
        context.submitNative(std::move(submission));
    }

    void detail::NativeAccess::invokeBlocking(MpiContext& context, detail::NativeBlockingSubmission submission)
    {
        context.invokeBlocking(std::move(submission));
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

                    MpiContext context{std::make_unique<MpiContext::Impl>()};
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
