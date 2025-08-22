#pragma once
#ifndef CRC_HPP
#define CRC_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace core {

template<typename T, typename P>
class CRC
{
public:
    ~CRC() noexcept = default;

    void update(const uint8_t *data, size_t size) noexcept
    {
        static_cast<P *>(this)->update(data, size);
    }

    T finalize() const noexcept
    {
        return _crc;
    }

protected:
    explicit constexpr CRC(T crc) noexcept : _crc(crc) { }

    T _crc;
};

} // namespace core

#endif // CRC_HPP
