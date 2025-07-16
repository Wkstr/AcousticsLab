#pragma once
#ifndef RING_BUFFER_HPP
#define RING_BUFFER_HPP

#include "logger.hpp"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace core {

template<typename T, std::enable_if_t<std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>, int> = 0>
class RingBuffer final
{
public:
    template<typename P = std::unique_ptr<RingBuffer<T>>>
    static P create(size_t max_capacity, void *buffer = nullptr, size_t size = 0)
    {
        if (max_capacity < 32)
        {
            LOG(ERROR, "RingBuffer max capacity must be at least 32");
            return nullptr;
        }

        if (size && !buffer)
        {
            LOG(ERROR, "Buffer cannot be null if size is provided");
            return nullptr;
        }

        if (max_capacity >= std::numeric_limits<size_t>::max() / sizeof(T))
        {
            LOG(ERROR, "RingBuffer max capacity is too large");
            return nullptr;
        }

        if (max_capacity % 32 != 0)
        {
            LOG(ERROR, "RingBuffer max capacity must be a multiple of 32");
            return nullptr;
        }

        const size_t bytes_needed = max_capacity * sizeof(T);
        if (size && size < bytes_needed)
        {
            LOG(ERROR, "Provided size (%zu) is less than required size (%zu)", size, bytes_needed);
            return nullptr;
        }

        bool internal_buffer = false;
        if (!buffer)
        {
            buffer = reinterpret_cast<T *>(new (std::nothrow) std::byte[bytes_needed]);
            if (!buffer)
            {
                LOG(ERROR, "Failed to allocate memory for RingBuffer, size: %zu bytes", bytes_needed);
                return nullptr;
            }
            internal_buffer = true;
        }

        P ptr(new (std::nothrow) RingBuffer<T>(internal_buffer, buffer, max_capacity));
        if (!ptr)
        {
            LOG(ERROR, "Failed to create RingBuffer, size: %zu bytes", bytes_needed);
            if (internal_buffer)
            {
                delete[] reinterpret_cast<std::byte *>(buffer);
            }
            return nullptr;
        }

        return ptr;
    }

    ~RingBuffer()
    {
        if (_internal_buffer && _buffer)
        {
            delete[] reinterpret_cast<std::byte *>(_buffer);
        }
        _buffer = nullptr;
    }

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer &operator=(const RingBuffer &) = delete;

    RingBuffer(RingBuffer &&other) = delete;
    RingBuffer &operator=(RingBuffer &&other) = delete;

    inline size_t size() const noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_relaxed);
        if (head >= tail)
        {
            return head - tail;
        }
        return _max_capacity - (tail - head);
    }

    inline size_t capacity() const noexcept
    {
        return _capacity_mask;
    }

    inline bool empty() const noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_relaxed);
        return head == tail;
    }

    const T *data() const noexcept
    {
        return _buffer;
    }

    inline size_t put(const T &value) noexcept
    {
        size_t head = _head.load(std::memory_order_relaxed);
        size_t tail = _tail.load(std::memory_order_acquire);

        const size_t next_head = (head + 1) & _capacity_mask;
        size_t next_tail = tail;
        const bool overflow = next_head == tail;

        T *ptr = _buffer + head;
        if (overflow) [[unlikely]]
        {
            next_tail = (tail + 1) & _capacity_mask;

            while (1)
            {
                void *expected_lock = nullptr;
                if (_lock.compare_exchange_weak(expected_lock, ptr, std::memory_order_acquire,
                        std::memory_order_relaxed)) [[likely]]
                {
                    break;
                }
            }
        }

        *ptr = value;

        _head.compare_exchange_strong(head, next_head, std::memory_order_release, std::memory_order_relaxed);
        if (overflow) [[unlikely]]
        {
            _lock.store(nullptr, std::memory_order_release);
            _tail.compare_exchange_strong(tail, next_tail, std::memory_order_release, std::memory_order_relaxed);
        }

        return 1;
    }

    inline size_t get(T &value) noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        size_t tail = _tail.load(std::memory_order_relaxed);
        if (head == tail) [[unlikely]]
        {
            return 0;
        }

        T *ptr = _buffer + tail;

        while (1)
        {
            void *expected_lock = nullptr;
            if (_lock.compare_exchange_weak(expected_lock, ptr, std::memory_order_acquire, std::memory_order_relaxed))
                [[likely]]
            {
                break;
            }
        }

        value = *ptr;

        _lock.store(nullptr, std::memory_order_release);

        const size_t next_tail = (tail + 1) & _capacity_mask;
        _tail.compare_exchange_strong(tail, next_tail, std::memory_order_release, std::memory_order_relaxed);

        _pos = 0;

        return 1;
    }

    inline size_t write(const T *buffer, size_t count) noexcept
    {
        if (!buffer || !count) [[unlikely]]
        {
            return 0;
        }

        const size_t rb_capacity = capacity();
        if (count > rb_capacity) [[unlikely]]
        {
            buffer += count - rb_capacity;
            count = rb_capacity;
        }

        size_t head = _head.load(std::memory_order_relaxed);
        size_t tail = _tail.load(std::memory_order_acquire);

        size_t available = head >= tail ? rb_capacity - head + tail : tail - head - 1;
        const size_t safe_count = std::min(count, available);

        const size_t part_1_start = head;
        const size_t part_1_end = std::min(part_1_start + safe_count, rb_capacity);
        const size_t part_1_count = part_1_end - part_1_start;
        if (part_1_count)
        {
            std::memcpy(_buffer + part_1_start, buffer, part_1_count * sizeof(T));
        }
        const size_t part_2_count = safe_count - part_1_count;
        if (part_2_count)
        {
            std::memcpy(_buffer, buffer + part_1_count, part_2_count * sizeof(T));
        }
        size_t written = safe_count;

        const size_t next_head = (head + safe_count) & _capacity_mask;
        _head.compare_exchange_strong(head, next_head, std::memory_order_release, std::memory_order_relaxed);

        while (written < count)
        {
            written += put(*(buffer + written));
        }

        return written;
    }

    inline size_t read(T *buffer, size_t count) noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        size_t tail = _tail.load(std::memory_order_relaxed);

        if (head == tail || !count) [[unlikely]]
        {
            return 0;
        }

        size_t copied = 0;
        size_t next_tail = tail;
        if (buffer)
        {
            while (head != next_tail && copied < count)
            {
                T *ptr = _buffer + next_tail;

                while (1)
                {
                    void *expected_lock = nullptr;
                    if (_lock.compare_exchange_weak(expected_lock, ptr, std::memory_order_acquire,
                            std::memory_order_relaxed)) [[likely]]
                    {
                        break;
                    }
                }

                buffer[copied++] = *ptr;

                _lock.store(nullptr, std::memory_order_release);

                next_tail = (next_tail + 1) & _capacity_mask;
            }
        }
        else
        {
            while (head != next_tail && copied < count)
            {
                ++copied;
                next_tail = (next_tail + 1) & _capacity_mask;
            }
        }

        size_t expected = tail;
        while (expected != next_tail)
        {
            if (_tail.compare_exchange_weak(expected, next_tail, std::memory_order_release, std::memory_order_relaxed))
            {
                break;
            }
            expected = ++tail & _capacity_mask;
        }

        _pos = 0;

        return copied;
    }

    inline size_t tellg() const noexcept
    {
        return _pos;
    }

    inline bool seekg(size_t pos) const noexcept
    {
        if (pos >= size()) [[unlikely]]
        {
            return false;
        }

        _pos = pos;

        return true;
    }

    inline bool peek(T &value) const noexcept
    {
        const size_t head = _head.load(std::memory_order_acquire);
        const size_t tail = _tail.load(std::memory_order_relaxed);
        const size_t size = head >= tail ? head - tail : _max_capacity - tail + head;
        if (_pos >= size) [[unlikely]]
        {
            return false;
        }

        const size_t pos = (tail + _pos++) & _capacity_mask;
        T *ptr = _buffer + pos;

        void *expected_lock = nullptr;
        while (1)
        {
            if (_lock.compare_exchange_weak(expected_lock, ptr, std::memory_order_acquire, std::memory_order_relaxed))
                [[likely]]
            {
                break;
            }
        }

        value = *ptr;

        _lock.store(nullptr, std::memory_order_release);

        return true;
    }

    void clear() noexcept
    {
        _head.store(0, std::memory_order_release);
        _tail.store(0, std::memory_order_release);
        _pos = 0;
    }

protected:
    RingBuffer(bool internal_buffer, void *buffer, size_t max_capacity)
        : _internal_buffer(internal_buffer), _buffer(static_cast<T *>(buffer)), _max_capacity(max_capacity),
          _capacity_mask(_max_capacity - 1), _head(0), _tail(0), _lock(nullptr), _pos(0)
    {
#ifdef __cpp_lib_atomic_is_always_lock_free
        if (!_head.is_always_lock_free || !_tail.is_always_lock_free)
        {
            LOG(WARNING, "RingBuffer head or tail is not lock-free, performance may be affected");
        }

        if (!_lock.is_always_lock_free)
        {
            LOG(WARNING, "RingBuffer lock is not lock-free, performance may be affected");
        }
#endif
    }

private:
    const bool _internal_buffer;
    T *_buffer;
    const size_t _max_capacity;
    const size_t _capacity_mask;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _head;
    alignas(std::hardware_destructive_interference_size) std::atomic<size_t> _tail;
    alignas(std::hardware_destructive_interference_size) std::atomic<void *> mutable _lock;
    alignas(std::hardware_destructive_interference_size) mutable size_t _pos;
#pragma GCC diagnostic pop
};

} // namespace core

#endif // RING_BUFFER_HPP
