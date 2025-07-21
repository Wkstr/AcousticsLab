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
    int encode(const U *data, size_t size, V &&write_callback) noexcept
    {
        return size <= _safe_size ? static_cast<P *>(this)->encode(data, size, std::forward<V>(write_callback))
                                  : -ERANGE;
    }

protected:
    Encoder(size_t safe_size) noexcept : _safe_size(safe_size) { }

private:
    const size_t _safe_size;
};

} // namespace core

#endif
