#pragma once
#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <type_traits>

namespace core {

template<typename T, typename P>
class Encoder
{
public:
    using ValueType = typename T::ValueType;
    using WriteCallback = typename T::WriteCallback;
    using StateType = typename T::State;

    ~Encoder() noexcept = default;

    StateType state() const noexcept
    {
        return static_cast<const P *>(this)->state();
    }

    template<typename U, std::enable_if_t<std::is_same_v<U, ValueType>, bool> = true>
    int encode(const U *data, size_t size, WriteCallback write_callback) noexcept
    {
        return static_cast<P *>(this)->encode(data, size, write_callback);
    }

protected:
    Encoder() noexcept = default;
};

} // namespace core

#endif
