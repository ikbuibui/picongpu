/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */
#include <algorithm>
#include <cassert>
#include <climits>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <caravan/mpi_native.hpp>
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

        bool validBuffer(BufferLease const& buffer)
        {
            return buffer.valid() && buffer.bytes() <= static_cast<std::size_t>(INT_MAX);
        }

        std::size_t scalarSize(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::int32:
            case ScalarType::uint32:
            case ScalarType::float32:
                return 4u;
            case ScalarType::int64:
            case ScalarType::uint64:
            case ScalarType::float64:
                return 8u;
            }
            return 0u;
        }

        MPI_Datatype nativeType(ScalarType type)
        {
            switch(type)
            {
            case ScalarType::int32:
                return MPI_INT32_T;
            case ScalarType::uint32:
                return MPI_UINT32_T;
            case ScalarType::int64:
                return MPI_INT64_T;
            case ScalarType::uint64:
                return MPI_UINT64_T;
            case ScalarType::float32:
                return MPI_FLOAT;
            case ScalarType::float64:
                return MPI_DOUBLE;
            }
            throw std::invalid_argument("Unknown Caravan scalar type");
        }

        MPI_Op nativeOperation(ReduceOperation operation)
        {
            switch(operation)
            {
            case ReduceOperation::sum:
                return MPI_SUM;
            case ReduceOperation::minimum:
                return MPI_MIN;
            case ReduceOperation::maximum:
                return MPI_MAX;
            case ReduceOperation::product:
                return MPI_PROD;
            }
            throw std::invalid_argument("Unknown Caravan reduce operation");
        }
    } // namespace

    class MpiExecutor::Impl
    {
    public:
        Impl() : m_owner(std::this_thread::get_id()), m_continuations(*this)
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

        void submitNative(Event predecessor, detail::NativeSubmission submission)
        {
            if(detail::executorDepth != 0u)
            {
                submission.setFailed(
                    std::make_exception_ptr(std::logic_error("Recursive native MPI submission is not allowed")));
                return;
            }
            auto const collective = submission.collective;
            submitAfter(
                std::move(predecessor),
                submission,
                [this](detail::NativeSubmission output) { startNative(std::move(output)); },
                collective);
        }

        void invokeBlocking(Event predecessor, detail::NativeBlockingSubmission submission)
        {
            if(detail::executorDepth != 0u)
            {
                submission.setFailed(
                    std::make_exception_ptr(std::logic_error("Recursive native MPI submission is not allowed")));
                return;
            }
            auto const collective = submission.collective;
            submitAfter(
                std::move(predecessor),
                submission,
                [this](detail::NativeBlockingSubmission output) { startBlocking(std::move(output)); },
                collective);
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
                {
                    lock.unlock();
                    releaseCommunicators();
                    return;
                }
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

        struct NativeGroup
        {
            detail::NativeSubmission output;
            std::vector<MPI_Status> statuses;
            std::vector<std::shared_ptr<void>> lifetimes;
            std::size_t remaining;
            bool terminal = false;

            bool complete(NativeMpiContext& context, std::size_t index, MPI_Status const& status)
            {
                if(terminal)
                    return false;
                statuses[index] = status;
                if(--remaining != 0u)
                    return false;
                terminal = true;
                try
                {
                    output.completed(context, statuses);
                }
                catch(...)
                {
                    output.failed(std::current_exception());
                }
                return true;
            }

            bool fail(std::exception_ptr error)
            {
                if(terminal)
                    return false;
                terminal = true;
                output.failed(std::move(error));
                return true;
            }
        };

        struct NativeCompletion
        {
            std::shared_ptr<NativeGroup> group;
            std::size_t index;
        };

        struct CollectiveTicket
        {
            CommunicatorId communicator;
            std::size_t sequence;
        };

        template<typename T_Output, typename T_Start>
        void submitAfter(Event predecessor, T_Output output, T_Start&& start, std::optional<CommunicatorId> collective)
        {
            std::optional<CollectiveTicket> ticket;
            {
                std::lock_guard lock(m_queueMutex);
                if(!m_accepting)
                {
                    output.setFailed(std::make_exception_ptr(std::runtime_error("MPI executor is shutting down")));
                    return;
                }
                ++m_outstanding;
                if(collective)
                    ticket = CollectiveTicket{*collective, m_collectiveSubmitted[collective->value]++};
            }

            predecessor.continueWith(
                m_continuations,
                [this, predecessor, output, start = std::forward<T_Start>(start), ticket](Event) mutable
                {
                    auto ready = [this, predecessor, output, start = std::move(start)]() mutable
                    {
                        if(predecessor.state() == CompletionState::failed)
                        {
                            output.setFailed(predecessor.error());
                            finishOperation();
                        }
                        else if(predecessor.state() == CompletionState::stopped)
                        {
                            output.setStopped();
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
                    };
                    if(ticket)
                        startCollective(*ticket, std::move(ready));
                    else
                        ready();
                });
        }

        void startCollective(CollectiveTicket ticket, std::function<void()> start)
        {
            assertOwner();
            auto& pending = m_collectivePending[ticket.communicator.value];
            pending.emplace(ticket.sequence, std::move(start));
            auto& next = m_collectiveNext[ticket.communicator.value];
            for(;;)
            {
                auto const found = pending.find(next);
                if(found == pending.end())
                    return;
                auto ready = std::move(found->second);
                pending.erase(found);
                ++next;
                ready();
            }
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
                    if(group->complete(context, index, MPI_Status{}))
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

        void failActive(NativeCompletion& active, std::exception_ptr failure)
        {
            if(active.group->fail(std::move(failure)))
                finishOperation();
        }

        void completeActive(NativeCompletion& active, MPI_Status const& status)
        {
            auto context = nativeContext();
            if(active.group->complete(context, active.index, status))
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
        std::unordered_map<std::uint32_t, std::size_t> m_collectiveSubmitted;
        std::unordered_map<std::uint32_t, std::size_t> m_collectiveNext;
        std::unordered_map<std::uint32_t, std::unordered_map<std::size_t, std::function<void()>>> m_collectivePending;
        std::vector<MPI_Comm> m_communicators{MPI_COMM_WORLD};
        std::vector<MPI_Request> m_requests;
        std::vector<NativeCompletion> m_active;
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

    TopologySnapshot detail::createCartesian(
        NativeMpiContext& context,
        std::vector<int> dimensions,
        std::vector<bool> periodic,
        int worldSize,
        int hostLocalRank)
    {
        std::size_t ranks = 1u;
        for(int dimension : dimensions)
        {
            if(dimension <= 0 || ranks > static_cast<std::size_t>(worldSize) / static_cast<std::size_t>(dimension))
                throw std::invalid_argument("Invalid Cartesian topology dimensions");
            ranks *= static_cast<std::size_t>(dimension);
        }
        if(dimensions.empty() || dimensions.size() != periodic.size() || ranks != static_cast<std::size_t>(worldSize))
            throw std::invalid_argument("Invalid Cartesian topology dimensions");

        std::vector<int> periods;
        periods.reserve(periodic.size());
        for(bool value : periodic)
            periods.emplace_back(value ? 1 : 0);

        TopologySnapshot snapshot;
        snapshot.hostLocalRank = hostLocalRank;
        snapshot.dimensions = std::move(dimensions);
        snapshot.periodic = std::move(periodic);
        snapshot.coordinates.resize(snapshot.dimensions.size());
        snapshot.neighbors.reserve(snapshot.dimensions.size() * 2u);

        MPI_Comm cartesian = MPI_COMM_NULL;
        int error = MPI_Cart_create(
            context.communicator(worldCommunicator),
            static_cast<int>(snapshot.dimensions.size()),
            snapshot.dimensions.data(),
            periods.data(),
            0,
            &cartesian);
        if(error != MPI_SUCCESS || cartesian == MPI_COMM_NULL)
            throw error == MPI_SUCCESS ? std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL")
                                       : mpiError("MPI_Cart_create", error);

        error = MPI_Comm_rank(cartesian, &snapshot.rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(cartesian, &snapshot.size);
        if(error == MPI_SUCCESS)
            error = MPI_Cart_coords(
                cartesian,
                snapshot.rank,
                static_cast<int>(snapshot.coordinates.size()),
                snapshot.coordinates.data());
        for(int dimension = 0; error == MPI_SUCCESS && dimension < static_cast<int>(snapshot.dimensions.size());
            ++dimension)
        {
            int negative = MPI_PROC_NULL;
            int positive = MPI_PROC_NULL;
            error = MPI_Cart_shift(cartesian, dimension, 1, &negative, &positive);
            snapshot.neighbors.emplace_back(negative == MPI_PROC_NULL ? -1 : negative);
            snapshot.neighbors.emplace_back(positive == MPI_PROC_NULL ? -1 : positive);
        }
        if(error != MPI_SUCCESS)
        {
            MPI_Comm_free(&cartesian);
            throw mpiError("MPI Cartesian topology query", error);
        }

        snapshot.communicator = context.adoptCommunicator(cartesian);
        return snapshot;
    }

    CommunicatorId detail::duplicateCommunicator(NativeMpiContext& context, CommunicatorId communicator)
    {
        MPI_Comm duplicate = MPI_COMM_NULL;
        int const error = MPI_Comm_dup(context.communicator(communicator), &duplicate);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Comm_dup", error);
        return context.adoptCommunicator(duplicate);
    }

    std::optional<CommunicatorInfo> detail::splitCommunicator(
        NativeMpiContext& context,
        std::optional<int> color,
        int key,
        CommunicatorId communicator)
    {
        MPI_Comm split = MPI_COMM_NULL;
        int error = MPI_Comm_split(context.communicator(communicator), color.value_or(MPI_UNDEFINED), key, &split);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Comm_split", error);
        if(split == MPI_COMM_NULL)
            return std::nullopt;

        CommunicatorInfo info;
        error = MPI_Comm_rank(split, &info.rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(split, &info.size);
        if(error != MPI_SUCCESS)
        {
            MPI_Comm_free(&split);
            throw mpiError("MPI split communicator query", error);
        }
        info.communicator = context.adoptCommunicator(split);
        return info;
    }

    void detail::destroyCommunicator(NativeMpiContext& context, CommunicatorId communicator)
    {
        if(communicator == worldCommunicator)
            throw std::invalid_argument("The world communicator cannot be destroyed");
        context.destroyCommunicator(communicator);
    }

    NativeRequestBatch detail::startSend(
        NativeMpiContext& context,
        BufferLease const& buffer,
        Peer destination,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || destination.any || destination.value < 0 || tag.any || tag.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI send");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.lifetime()});
        int const error = MPI_Isend(
            buffer.data(),
            static_cast<int>(buffer.bytes()),
            MPI_BYTE,
            destination.value,
            tag.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Isend", error);
        return batch;
    }

    NativeRequestBatch detail::startReceive(
        NativeMpiContext& context,
        BufferLease const& buffer,
        Peer source,
        MessageTag tag,
        CommunicatorId communicator)
    {
        if(!validBuffer(buffer) || (!source.any && source.value < 0) || (!tag.any && tag.value < 0))
            throw std::invalid_argument("Invalid Caravan MPI receive");

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {buffer.lifetime()});
        int const error = MPI_Irecv(
            buffer.data(),
            static_cast<int>(buffer.bytes()),
            MPI_BYTE,
            source.any ? MPI_ANY_SOURCE : source.value,
            tag.any ? MPI_ANY_TAG : tag.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Irecv", error);
        return batch;
    }

    ReceiveResult detail::completeReceive(std::span<MPI_Status const> statuses)
    {
        int bytes = 0;
        int const error = MPI_Get_count(&statuses.front(), MPI_BYTE, &bytes);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Get_count", error);
        if(bytes == MPI_UNDEFINED)
            throw std::runtime_error("MPI_Get_count returned MPI_UNDEFINED");
        return ReceiveResult{
            Peer{statuses.front().MPI_SOURCE},
            MessageTag{statuses.front().MPI_TAG},
            static_cast<std::size_t>(bytes)};
    }

    NativeRequestBatch detail::startAllReduce(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& elements)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output) || input.bytes() % elementBytes != 0u
           || output.bytes() < input.bytes())
            throw std::invalid_argument("Invalid Caravan MPI all-reduce");
        *elements = input.bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.lifetime(), output.lifetime(), elements});
        void* const sendBuffer = input.data() == output.data() ? MPI_IN_PLACE : input.data();
        int const error = MPI_Iallreduce(
            sendBuffer,
            output.data(),
            static_cast<int>(*elements),
            nativeType(type),
            nativeOperation(operation),
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Iallreduce", error);
        return batch;
    }

    Future<AllReduceResult> MpiExecutor::allReduce(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator)
    {
        auto elements = std::make_shared<std::size_t>(0u);
        return nativeFuture<AllReduceResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input), output = std::move(output), type, operation, communicator, elements](
                NativeMpiContext& context)
            { return detail::startAllReduce(context, input, output, type, operation, communicator, elements); },
            [elements](std::span<MPI_Status const>) { return AllReduceResult{*elements}; },
            communicator);
    }

    NativeRequestBatch detail::startReduce(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& elements)
    {
        auto const elementBytes = scalarSize(type);
        if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output) || input.bytes() % elementBytes != 0u
           || output.bytes() < input.bytes() || input.data() == output.data() || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI reduce");
        *elements = input.bytes() / elementBytes;

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.lifetime(), output.lifetime(), elements});
        int const error = MPI_Ireduce(
            input.data(),
            output.data(),
            static_cast<int>(*elements),
            nativeType(type),
            nativeOperation(operation),
            root.value,
            context.communicator(communicator),
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ireduce", error);
        return batch;
    }

    Future<ReduceResult> MpiExecutor::reduce(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        Peer root,
        CommunicatorId communicator)
    {
        auto elements = std::make_shared<std::size_t>(0u);
        return nativeFuture<ReduceResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input), output = std::move(output), type, operation, root, communicator, elements](
                NativeMpiContext& context)
            { return detail::startReduce(context, input, output, type, operation, root, communicator, elements); },
            [elements](std::span<MPI_Status const>) { return ReduceResult{*elements}; },
            communicator);
    }

    NativeRequestBatch detail::startGather(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || root.any || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI gather");

        auto const native = context.communicator(communicator);
        int rank = -1;
        int size = 0;
        int error = MPI_Comm_rank(native, &rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI gather communicator query", error);
        if(rank == root.value)
        {
            if(size <= 0 || input.bytes() > output.bytes() / static_cast<std::size_t>(size))
                throw std::invalid_argument("Caravan MPI gather output is too small");
            *resultBytes = input.bytes() * static_cast<std::size_t>(size);
        }

        NativeRequestBatch batch({MPI_REQUEST_NULL}, {input.lifetime(), output.lifetime(), resultBytes});
        error = MPI_Igather(
            input.data(),
            static_cast<int>(input.bytes()),
            MPI_BYTE,
            output.data(),
            static_cast<int>(input.bytes()),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igather", error);
        return batch;
    }

    Future<GatherResult> MpiExecutor::gather(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        Peer root,
        CommunicatorId communicator)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        return nativeFuture<GatherResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input), output = std::move(output), root, communicator, resultBytes](
                NativeMpiContext& context)
            { return detail::startGather(context, input, output, root, communicator, resultBytes); },
            [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
            communicator);
    }

    NativeRequestBatch detail::startGatherV(
        NativeMpiContext& context,
        BufferLease const& input,
        BufferLease const& output,
        std::vector<std::size_t> const& receiveBytes,
        std::vector<std::size_t> const& displacements,
        Peer root,
        CommunicatorId communicator,
        std::shared_ptr<std::size_t> const& resultBytes)
    {
        if(!validBuffer(input) || !validBuffer(output) || receiveBytes.size() != displacements.size() || root.any
           || root.value < 0)
            throw std::invalid_argument("Invalid Caravan MPI variable gather");

        auto counts = std::make_shared<std::vector<int>>();
        auto offsets = std::make_shared<std::vector<int>>();
        counts->reserve(receiveBytes.size());
        offsets->reserve(displacements.size());
        for(std::size_t i = 0u; i < receiveBytes.size(); ++i)
        {
            if(receiveBytes[i] > static_cast<std::size_t>(INT_MAX)
               || displacements[i] > static_cast<std::size_t>(INT_MAX))
                throw std::invalid_argument("Invalid Caravan MPI variable gather layout");
            counts->emplace_back(static_cast<int>(receiveBytes[i]));
            offsets->emplace_back(static_cast<int>(displacements[i]));
        }

        auto const native = context.communicator(communicator);
        int rank = -1;
        int size = 0;
        int error = MPI_Comm_rank(native, &rank);
        if(error == MPI_SUCCESS)
            error = MPI_Comm_size(native, &size);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI variable gather communicator query", error);
        if(rank == root.value)
        {
            if(size <= 0 || counts->size() != static_cast<std::size_t>(size))
                throw std::invalid_argument("Invalid Caravan MPI variable gather rank count");
            for(std::size_t i = 0u; i < counts->size(); ++i)
            {
                auto const end = static_cast<std::size_t>((*offsets)[i]) + static_cast<std::size_t>((*counts)[i]);
                if(end > output.bytes())
                    throw std::invalid_argument("Caravan MPI variable gather output is too small");
                *resultBytes += static_cast<std::size_t>((*counts)[i]);
            }
        }

        NativeRequestBatch batch(
            {MPI_REQUEST_NULL},
            {input.lifetime(), output.lifetime(), counts, offsets, resultBytes});
        error = MPI_Igatherv(
            input.data(),
            static_cast<int>(input.bytes()),
            MPI_BYTE,
            output.data(),
            counts->data(),
            offsets->data(),
            MPI_BYTE,
            root.value,
            native,
            &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Igatherv", error);
        return batch;
    }

    Future<GatherResult> MpiExecutor::gatherV(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        std::vector<std::size_t> receiveBytes,
        std::vector<std::size_t> displacements,
        Peer root,
        CommunicatorId communicator)
    {
        auto resultBytes = std::make_shared<std::size_t>(0u);
        return nativeFuture<GatherResult>(
            *this,
            std::move(dataReady),
            [input = std::move(input),
             output = std::move(output),
             receiveBytes = std::move(receiveBytes),
             displacements = std::move(displacements),
             root,
             communicator,
             resultBytes](NativeMpiContext& context)
            {
                return detail::startGatherV(
                    context,
                    input,
                    output,
                    receiveBytes,
                    displacements,
                    root,
                    communicator,
                    resultBytes);
            },
            [resultBytes](std::span<MPI_Status const>) { return GatherResult{*resultBytes}; },
            communicator);
    }

    NativeRequestBatch detail::startBarrier(NativeMpiContext& context, CommunicatorId communicator)
    {
        NativeRequestBatch batch({MPI_REQUEST_NULL});
        int const error = MPI_Ibarrier(context.communicator(communicator), &batch.requests[0]);
        if(error != MPI_SUCCESS)
            throw mpiError("MPI_Ibarrier", error);
        return batch;
    }

    Event MpiExecutor::barrier(Event predecessor, CommunicatorId communicator)
    {
        return nativeEvent(
            *this,
            std::move(predecessor),
            [communicator](NativeMpiContext& context) { return detail::startBarrier(context, communicator); },
            [](std::span<MPI_Status const>) {},
            communicator);
    }

    void MpiExecutor::run()
    {
        m_implementation->run();
    }

    void MpiExecutor::requestShutdown()
    {
        m_implementation->requestShutdown();
    }

    void MpiExecutor::submitNative(Event predecessor, detail::NativeSubmission submission)
    {
        m_implementation->submitNative(std::move(predecessor), std::move(submission));
    }

    void MpiExecutor::invokeBlocking(Event predecessor, detail::NativeBlockingSubmission submission)
    {
        m_implementation->invokeBlocking(std::move(predecessor), std::move(submission));
    }

    void detail::NativeAccess::submit(MpiExecutor& executor, Event predecessor, detail::NativeSubmission submission)
    {
        executor.submitNative(std::move(predecessor), std::move(submission));
    }

    void detail::NativeAccess::invokeBlocking(
        MpiExecutor& executor,
        Event predecessor,
        detail::NativeBlockingSubmission submission)
    {
        executor.invokeBlocking(std::move(predecessor), std::move(submission));
    }

    int MpiRuntime::runImpl(int& argc, char**& argv, std::function<int(MpiExecutor&)> application)
    {
        std::promise<MpiExecutor*> startup;
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

                    MpiExecutor executor{std::make_unique<MpiExecutor::Impl>()};
                    startup.set_value(&executor);
                    published = true;
                    executor.run();

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

        MpiExecutor* executor;
        try
        {
            executor = ready.get();
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
            applicationResult = application(*executor);
        }
        catch(...)
        {
            applicationError = std::current_exception();
        }
        executor->requestShutdown();
        mpiWorker.join();

        if(applicationError)
            std::rethrow_exception(applicationError);
        if(workerError)
            std::rethrow_exception(workerError);
        return applicationResult;
    }
} // namespace caravan
