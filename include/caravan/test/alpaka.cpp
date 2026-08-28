/*
 * This file is part of PIConGPU.
 * SPDX-License-Identifier: GPL-3.0-or-later OR LGPL-3.0-or-later
 */

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>

#include <caravan/alpaka.hpp>
#include <caravan/core.hpp>

namespace
{
    struct Increment
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(T_Acc const&, int* value) const
        {
            ++*value;
        }
    };
} // namespace

int main()
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::Queue<Acc, alpaka::NonBlocking>;

    auto const device = alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0u);
    auto const host = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0u);
    Queue queue{device};
    auto deviceValue = alpaka::allocBuf<int, Idx>(device, alpaka::Vec<Dim, Idx>{1u});
    auto hostValue = alpaka::allocBuf<int, Idx>(host, alpaka::Vec<Dim, Idx>{1u});
    hostValue[0] = 41;

    bool submitted = false;
    auto retained = std::make_shared<int>(7);
    std::weak_ptr<int> retainedObserver = retained;
    auto sender = caravan::alpaka::submit(
        queue,
        [&, retained = std::move(retained)](Queue& nativeQueue)
        {
            submitted = true;
            alpaka::memset(nativeQueue, deviceValue, 0);
            alpaka::memcpy(nativeQueue, deviceValue, hostValue);
            alpaka::exec<Acc>(
                nativeQueue,
                alpaka::WorkDivMembers<Dim, Idx>{
                    alpaka::Vec<Dim, Idx>{1u},
                    alpaka::Vec<Dim, Idx>{1u},
                    alpaka::Vec<Dim, Idx>{1u}},
                Increment{},
                alpaka::getPtrNative(deviceValue));
            alpaka::memcpy(nativeQueue, hostValue, deviceValue);
            assert(*retained == 7);
        });

    static_assert(caravan::Sender<decltype(sender)>);
    assert(!submitted);
    assert(!retainedObserver.expired());

    caravan::RunLoop loop;
    caravan::AsyncScope scope;
    std::thread::id continuationThread;
    auto completion = scope.spawn(
        caravan::then(
            caravan::continuesOn(std::move(sender), loop.scheduler()),
            [&] { continuationThread = std::this_thread::get_id(); }));
    assert(submitted);

    while(completion.state() == caravan::CompletionState::pending)
    {
        loop.runReady();
        std::this_thread::yield();
    }
    completion.wait();

    auto failed
        = scope.spawn(caravan::alpaka::submit(queue, [](Queue&) { throw std::runtime_error("submission failed"); }));
    assert(failed.state() == caravan::CompletionState::failed);

    scope.join().wait();
    assert(hostValue[0] == 42);
    assert(continuationThread == std::this_thread::get_id());
    assert(retainedObserver.expired());
}
