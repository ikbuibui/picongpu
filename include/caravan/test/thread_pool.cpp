/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>

#include <caravan/core.hpp>

namespace
{
    thread_local bool failAllocation = false;

    struct Receiver
    {
        void set_value() noexcept
        {
            assert(false); // Executor initialization is forced to fail in this operation.
        }

        void set_error(std::exception_ptr error) noexcept
        {
            *failure = std::move(error);
        }

        std::exception_ptr* failure;
    };
} // namespace

void* operator new(std::size_t bytes)
{
    if(failAllocation)
    {
        failAllocation = false;
        throw std::bad_alloc{};
    }
    if(auto* memory = std::malloc(bytes == 0u ? 1u : bytes))
        return memory;
    throw std::bad_alloc{};
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    if(std::string_view{argv[1]} == "invalid")
    {
        // Exercise the actual schedule sender, not just the configuration parser.
        bool failed = false;
        try
        {
            caravan::syncWait(caravan::SubmissionScheduler{}.schedule());
        }
        catch(std::invalid_argument const&)
        {
            failed = true;
        }
        assert(failed);
        return 0;
    }

    // Failed singleton initialization must deliver an error and allow a subsequent successful retry.
    std::exception_ptr failure;
    auto operation = caravan::SubmissionScheduler{}.schedule().connect(Receiver{&failure});
    failAllocation = true;
    operation.start();
    assert(failure);
    try
    {
        std::rethrow_exception(failure);
    }
    catch(std::bad_alloc const&)
    {
    }

    auto const workerCount = static_cast<std::size_t>(std::strtoul(argv[1], nullptr, 10));
    auto const caller = std::this_thread::get_id();
    std::atomic<unsigned> completed = 0u;
    caravan::AsyncScope scope;
    for(unsigned i = 0u; i < 1000u; ++i)
        scope.spawn(
            caravan::SubmissionScheduler{}.schedule()
            | caravan::then(
                [&]
                {
                    if(workerCount == 0u)
                        assert(std::this_thread::get_id() == caller);
                    else
                    {
                        assert(std::this_thread::get_id() != caller);
                        assert(caravan::isExecutorThread());
                    }
                    ++completed;
                }));
    scope.join().wait();
    assert(completed == 1000u);
    assert(caravan::submissionExecutor().threadCount() == workerCount);

    // The public pool rejects zero workers, and destruction drains all accepted tasks.
    bool rejected = false;
    try
    {
        caravan::ThreadPool empty{0u};
    }
    catch(std::invalid_argument const&)
    {
        rejected = true;
    }
    assert(rejected);
    completed = 0u;
    {
        caravan::ThreadPool pool{2u};
        for(unsigned i = 0u; i < 1000u; ++i)
            pool.post([&] { ++completed; });
    }
    assert(completed == 1000u);
}
