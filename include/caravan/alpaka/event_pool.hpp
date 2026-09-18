/*
 * This file is part of Caravan.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace caravan::alpaka::detail
{
    /** Device-local event storage. Leases retain the backing storage and must outlive native use. */
    template<typename T_Queue>
    class EventPool
    {
        using Event = ::alpaka::Event<T_Queue>;

        struct Entry
        {
            explicit Entry(::alpaka::Dev<T_Queue> const& device) : event(device)
            {
            }

            Event event;
            Entry* next = nullptr;
        };

        struct Storage
        {
            explicit Storage(::alpaka::Dev<T_Queue> device) : device(std::move(device))
            {
            }

            ::alpaka::Dev<T_Queue> device;
            std::mutex mutex;
            std::vector<std::unique_ptr<Entry>> entries;
            Entry* free = nullptr;
        };

    public:
        class Lease
        {
        public:
            Lease(Lease const&) = delete;
            Lease& operator=(Lease const&) = delete;

            Lease(Lease&& other) noexcept : m_storage(std::move(other.m_storage)), m_entry(other.m_entry)
            {
            }

            ~Lease()
            {
                if(m_storage)
                {
                    std::lock_guard lock(m_storage->mutex);
                    m_entry->next = m_storage->free;
                    m_storage->free = m_entry;
                }
            }

            Event const& event() const noexcept
            {
                return m_entry->event;
            }

        private:
            friend class EventPool;

            Lease(std::shared_ptr<Storage> storage, Entry& entry)
                : m_storage(std::move(storage))
                , m_entry(&entry)
            {
            }

            std::shared_ptr<Storage> m_storage;
            Entry* m_entry;
        };

        explicit EventPool(::alpaka::Dev<T_Queue> device) : m_storage(std::make_shared<Storage>(std::move(device)))
        {
        }

        EventPool(EventPool const&) = delete;
        EventPool& operator=(EventPool const&) = delete;

        Lease acquire()
        {
            std::lock_guard lock(m_storage->mutex);
            if(m_storage->free)
            {
                auto* entry = m_storage->free;
                m_storage->free = entry->next;
                return Lease{m_storage, *entry};
            }
            // Retain the high-water mark until the pool and all exported dependencies are released.
            m_storage->entries.push_back(std::make_unique<Entry>(m_storage->device));
            return Lease{m_storage, *m_storage->entries.back()};
        }

    private:
        std::shared_ptr<Storage> m_storage;
    };
} // namespace caravan::alpaka::detail
