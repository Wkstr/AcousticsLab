#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <cerrno>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace core {

template<typename T, typename P>
class Encoder: public T
{
public:
    ~Encoder() noexcept = default;

    T::State state() const noexcept
    {
        return static_cast<const P *>(this)->state();
    }

    template<typename U, typename V, std::enable_if_t<std::is_same_v<U, typename T::ValueType>, bool> = true>
    size_t encode(const U *data, size_t size, V &&write_callback) noexcept
    {
        if (size <= _safe_size) [[likely]]
        {
            return static_cast<P *>(this)->encode(data, size, std::forward<V>(write_callback));
        }
        _error = EOVERFLOW;
        return 0;
    }

    size_t estimate(size_t size) const noexcept
    {
        return static_cast<const P *>(this)->estimate(size);
    }

    int error() const noexcept
    {
        return _error;
    }

protected:
    Encoder(size_t safe_size) noexcept : _safe_size(safe_size), _error(0) { }

    const size_t _safe_size;
    int _error;
};

} // namespace core

#endif
