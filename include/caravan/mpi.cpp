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
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
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

        Future<TopologySnapshot> createCartesian(
            Event predecessor,
            std::vector<int> dimensions,
            std::vector<bool> periodic)
        {
            Promise<TopologySnapshot> completion;
            auto result = completion.future();
            std::size_t ranks = 1u;
            for(int dimension : dimensions)
            {
                if(dimension <= 0 || ranks > static_cast<std::size_t>(m_topology.size) / dimension)
                {
                    completion.setFailed(
                        std::make_exception_ptr(std::invalid_argument("Invalid Cartesian topology dimensions")));
                    return result;
                }
                ranks *= static_cast<std::size_t>(dimension);
            }
            if(dimensions.empty() || dimensions.size() != periodic.size()
               || ranks != static_cast<std::size_t>(m_topology.size))
            {
                completion.setFailed(
                    std::make_exception_ptr(std::invalid_argument("Invalid Cartesian topology dimensions")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, dimensions = std::move(dimensions), periodic = std::move(periodic)](
                    Promise<TopologySnapshot> output) mutable
                { startCreateCartesian(std::move(output), std::move(dimensions), std::move(periodic)); });
            return result;
        }

        Event destroyCommunicator(Event predecessor, CommunicatorId communicator)
        {
            EventSource completion;
            auto result = completion.event();
            if(communicator == worldCommunicator)
            {
                completion.setFailed(
                    std::make_exception_ptr(std::invalid_argument("The world communicator cannot be destroyed")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, communicator](EventSource output)
                { startDestroyCommunicator(std::move(output), communicator); });
            return result;
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
            if(!validBuffer(buffer) || destination.any || destination.value < 0 || tag.any || tag.value < 0)
            {
                completion.setFailed(std::make_exception_ptr(std::invalid_argument("Invalid Caravan MPI send")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, buffer = std::move(buffer), destination, tag, communicator](Promise<SendResult> output) mutable
                { startSend(std::move(output), std::move(buffer), destination, tag, communicator); });
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
            if(!validBuffer(buffer) || (!source.any && source.value < 0) || (!tag.any && tag.value < 0))
            {
                completion.setFailed(std::make_exception_ptr(std::invalid_argument("Invalid Caravan MPI receive")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, buffer = std::move(buffer), source, tag, communicator](Promise<ReceiveResult> output) mutable
                { startReceive(std::move(output), std::move(buffer), source, tag, communicator); });
            return result;
        }

        Future<AllReduceResult> allReduce(
            Event predecessor,
            BufferLease input,
            BufferLease output,
            ScalarType type,
            ReduceOperation operation,
            CommunicatorId communicator)
        {
            Promise<AllReduceResult> completion;
            auto result = completion.future();
            auto const elementBytes = scalarSize(type);
            if(elementBytes == 0u || !validBuffer(input) || !validBuffer(output) || input.bytes() % elementBytes != 0u
               || output.bytes() < input.bytes())
            {
                completion.setFailed(std::make_exception_ptr(std::invalid_argument("Invalid Caravan MPI all-reduce")));
                return result;
            }
            submitAfter(
                std::move(predecessor),
                completion,
                [this, input = std::move(input), output = std::move(output), type, operation, communicator](
                    Promise<AllReduceResult> result) mutable
                {
                    startAllReduce(
                        std::move(result),
                        std::move(input),
                        std::move(output),
                        type,
                        operation,
                        communicator);
                });
            return result;
        }

        Event barrier(Event predecessor, CommunicatorId communicator)
        {
            EventSource completion;
            auto result = completion.event();
            submitAfter(
                std::move(predecessor),
                completion,
                [this, communicator](EventSource output) { startBarrier(std::move(output), communicator); });
            return result;
        }

        void submitNative(Event predecessor, detail::NativeSubmission submission)
        {
            if(detail::executorDepth != 0u)
            {
                submission.setFailed(
                    std::make_exception_ptr(std::logic_error("Recursive native MPI submission is not allowed")));
                return;
            }
            submitAfter(
                std::move(predecessor),
                submission,
                [this](detail::NativeSubmission output) { startNative(std::move(output)); });
        }

        void invokeBlocking(Event predecessor, detail::NativeBlockingSubmission submission)
        {
            if(detail::executorDepth != 0u)
            {
                submission.setFailed(
                    std::make_exception_ptr(std::logic_error("Recursive native MPI submission is not allowed")));
                return;
            }
            submitAfter(
                std::move(predecessor),
                submission,
                [this](detail::NativeBlockingSubmission output) { startBlocking(std::move(output)); });
        }

        void run()
        {
            assertOwner();
            ExecutorThreadGuard guard;
            for(;;)
            {
                drainQueue();
                progress();
                if(m_requests.empty() && m_blocking)
                    runBlocking();

                std::unique_lock lock(m_queueMutex);
                if(m_stopping && m_outstanding == 0u)
                {
                    lock.unlock();
                    releaseCommunicators();
                    return;
                }
                if(m_requests.empty() && m_queue.empty() && !m_blocking)
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

        struct AllReduceCompletion
        {
            Promise<AllReduceResult> output;
            std::size_t elements;
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

        using Barrier = BarrierCompletion;
        using Send = SendCompletion;
        using Receive = ReceiveCompletion;
        using AllReduce = AllReduceCompletion;
        using Native = NativeCompletion;
        using ActiveCompletion = std::variant<Barrier, Send, Receive, AllReduce, Native>;

        struct ActiveOperation
        {
            ActiveCompletion completion;
            std::optional<BufferLease> firstBuffer;
            std::optional<BufferLease> secondBuffer;
        };

        static bool validBuffer(BufferLease const& buffer)
        {
            return buffer.valid() && buffer.bytes() <= static_cast<std::size_t>(INT_MAX);
        }

        static std::size_t scalarSize(ScalarType type)
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
            while(!m_blocking)
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

        void prepareActive()
        {
            auto const capacity = m_requests.size() + 1u;
            m_requests.reserve(capacity);
            m_active.reserve(capacity);
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

        NativeMpiContext nativeContext()
        {
            return detail::NativeContextFactory::create(
                this,
                [](void* implementation, CommunicatorId id)
                { return static_cast<Impl*>(implementation)->communicator(id); },
                [](void* implementation, MPI_Comm native)
                { return static_cast<Impl*>(implementation)->adoptCommunicator(native); });
        }

        void startNative(detail::NativeSubmission output)
        {
            assertOwner();
            auto context = nativeContext();
            auto batch = output.start(context);
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
                m_active.emplace_back(NativeCompletion{group, index}, std::nullopt, std::nullopt);
            }
            detail::NativeAccess::release(batch);
        }

        void startBlocking(detail::NativeBlockingSubmission output)
        {
            assertOwner();
            if(m_blocking)
                throw std::logic_error("Nested native blocking MPI operation");
            m_blocking.emplace(std::move(output));
            if(m_requests.empty())
                runBlocking();
        }

        void runBlocking()
        {
            assertOwner();
            auto output = std::move(*m_blocking);
            m_blocking.reset();
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

        void startCreateCartesian(
            Promise<TopologySnapshot> completion,
            std::vector<int> dimensions,
            std::vector<bool> periodic)
        {
            assertOwner();
            if(!m_requests.empty())
            {
                completion.setFailed(
                    std::make_exception_ptr(std::runtime_error("Communicator creation requires MPI quiescence")));
                finishOperation();
                return;
            }

            m_communicators.reserve(m_communicators.size() + 1u);
            std::vector<int> periods;
            periods.reserve(periodic.size());
            for(bool value : periodic)
                periods.emplace_back(value ? 1 : 0);

            MPI_Comm cartesian = MPI_COMM_NULL;
            int const createError = MPI_Cart_create(
                MPI_COMM_WORLD,
                static_cast<int>(dimensions.size()),
                dimensions.data(),
                periods.data(),
                0,
                &cartesian);
            if(createError != MPI_SUCCESS || cartesian == MPI_COMM_NULL)
            {
                completion.setFailed(
                    std::make_exception_ptr(
                        createError == MPI_SUCCESS ? std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL")
                                                   : mpiError("MPI_Cart_create", createError)));
                finishOperation();
                return;
            }

            TopologySnapshot snapshot;
            snapshot.hostLocalRank = m_topology.hostLocalRank;
            snapshot.communicator = CommunicatorId{static_cast<std::uint32_t>(m_communicators.size())};
            snapshot.dimensions = std::move(dimensions);
            snapshot.periodic = std::move(periodic);
            snapshot.coordinates.resize(snapshot.dimensions.size());
            snapshot.neighbors.reserve(snapshot.dimensions.size() * 2u);

            int error = MPI_Comm_rank(cartesian, &snapshot.rank);
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
                completion.setFailed(std::make_exception_ptr(mpiError("MPI Cartesian topology query", error)));
                finishOperation();
                return;
            }

            m_communicators.emplace_back(cartesian);
            completion.setValue(std::move(snapshot));
            finishOperation();
        }

        void startDestroyCommunicator(EventSource completion, CommunicatorId id)
        {
            assertOwner();
            if(!m_requests.empty())
            {
                completion.setFailed(
                    std::make_exception_ptr(std::runtime_error("Communicator destruction requires MPI quiescence")));
                finishOperation();
                return;
            }
            try
            {
                auto& native = m_communicators.at(id.value);
                if(native == MPI_COMM_NULL)
                    throw std::invalid_argument("Unknown Caravan communicator");
                int const error = MPI_Comm_free(&native);
                if(error != MPI_SUCCESS)
                    throw mpiError("MPI_Comm_free", error);
                completion.setReady();
            }
            catch(...)
            {
                completion.setFailed(std::current_exception());
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

        void startBarrier(EventSource completion, CommunicatorId communicatorId)
        {
            assertOwner();
            prepareActive();
            MPI_Request request = MPI_REQUEST_NULL;
            int const error = MPI_Ibarrier(communicator(communicatorId), &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Ibarrier", error)));
                finishOperation();
                return;
            }
            m_requests.emplace_back(request);
            m_active.emplace_back(BarrierCompletion{std::move(completion)}, std::nullopt, std::nullopt);
        }

        void startSend(
            Promise<SendResult> completion,
            BufferLease buffer,
            Peer destination,
            MessageTag tag,
            CommunicatorId communicatorId)
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
                communicator(communicatorId),
                &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Isend", error)));
                finishOperation();
                return;
            }
            auto const bytes = buffer.bytes();
            m_requests.emplace_back(request);
            m_active.emplace_back(SendCompletion{std::move(completion), bytes}, std::move(buffer), std::nullopt);
        }

        void startReceive(
            Promise<ReceiveResult> completion,
            BufferLease buffer,
            Peer source,
            MessageTag tag,
            CommunicatorId communicatorId)
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
                communicator(communicatorId),
                &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Irecv", error)));
                finishOperation();
                return;
            }
            m_requests.emplace_back(request);
            m_active.emplace_back(ReceiveCompletion{std::move(completion)}, std::move(buffer), std::nullopt);
        }

        static MPI_Datatype nativeType(ScalarType type)
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

        static MPI_Op nativeOperation(ReduceOperation operation)
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

        void startAllReduce(
            Promise<AllReduceResult> completion,
            BufferLease input,
            BufferLease output,
            ScalarType type,
            ReduceOperation operation,
            CommunicatorId communicatorId)
        {
            assertOwner();
            prepareActive();
            auto const elements = input.bytes() / scalarSize(type);
            MPI_Request request = MPI_REQUEST_NULL;
            void* const sendBuffer = input.data() == output.data() ? MPI_IN_PLACE : input.data();
            int const error = MPI_Iallreduce(
                sendBuffer,
                output.data(),
                static_cast<int>(elements),
                nativeType(type),
                nativeOperation(operation),
                communicator(communicatorId),
                &request);
            if(error != MPI_SUCCESS)
            {
                completion.setFailed(std::make_exception_ptr(mpiError("MPI_Iallreduce", error)));
                finishOperation();
                return;
            }
            m_requests.emplace_back(request);
            m_active.emplace_back(
                AllReduceCompletion{std::move(completion), elements},
                std::move(input),
                std::move(output));
        }

        void failActive(ActiveOperation& active, std::exception_ptr failure)
        {
            std::visit(
                [&](auto& completion)
                {
                    using Completion = std::remove_cvref_t<decltype(completion)>;
                    if constexpr(std::is_same_v<Completion, NativeCompletion>)
                    {
                        if(completion.group->fail(failure))
                            finishOperation();
                    }
                    else
                    {
                        completion.output.setFailed(failure);
                        finishOperation();
                    }
                },
                active.completion);
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
                    else if constexpr(std::is_same_v<Completion, ReceiveCompletion>)
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
                    else if constexpr(std::is_same_v<Completion, AllReduceCompletion>)
                        completion.output.setValue(AllReduceResult{completion.elements});
                    else
                    {
                        auto context = nativeContext();
                        if(completion.group->complete(context, completion.index, status))
                            finishOperation();
                    }
                },
                active.completion);
            if(!std::holds_alternative<NativeCompletion>(active.completion))
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
        std::vector<MPI_Comm> m_communicators{MPI_COMM_WORLD};
        std::vector<MPI_Request> m_requests;
        std::vector<ActiveOperation> m_active;
        std::vector<int> m_completedIndices;
        std::vector<MPI_Status> m_statuses;
        std::optional<detail::NativeBlockingSubmission> m_blocking;
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

    Future<TopologySnapshot> MpiExecutor::createCartesian(
        Event predecessor,
        std::vector<int> dimensions,
        std::vector<bool> periodic)
    {
        return m_implementation->createCartesian(std::move(predecessor), std::move(dimensions), std::move(periodic));
    }

    Event MpiExecutor::destroyCommunicator(Event predecessor, CommunicatorId communicator)
    {
        return m_implementation->destroyCommunicator(std::move(predecessor), communicator);
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

    Future<AllReduceResult> MpiExecutor::allReduce(
        Event dataReady,
        BufferLease input,
        BufferLease output,
        ScalarType type,
        ReduceOperation operation,
        CommunicatorId communicator)
    {
        return m_implementation
            ->allReduce(std::move(dataReady), std::move(input), std::move(output), type, operation, communicator);
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
