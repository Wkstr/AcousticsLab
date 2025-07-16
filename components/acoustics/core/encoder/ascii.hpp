#pragma once
#ifndef ASCII_HPP
#define ASCII_HPP

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <new>

#include "core/encoder.hpp"

namespace core {

namespace encoder {

    class ASCII
    {
    public:
        using ValueType = unsigned char;
        using WriteCallback = std::function<int(const void *data, size_t size)>;

        struct State final
        {
            State() noexcept = default;

            ~State() noexcept = default;
        };
    };

    namespace detail {

        class ASCIIBase64 final: public Encoder<ASCII, ASCIIBase64>
        {
        public:
            ASCIIBase64(void *buffer = nullptr, size_t buffer_size = 1024) noexcept
                : _state(), _buffer(buffer), _buffer_size(buffer_size), _internal_buffer(false)
            {
                if (!_buffer && _buffer_size >= 4)
                {
                    _buffer = new std::byte[_buffer_size];
                    _internal_buffer = true;
                }
                else if (_buffer_size < 4) [[unlikely]]
                {
                    _buffer = nullptr;
                    _buffer_size = 0;
                }
            }

            ~ASCIIBase64() noexcept
            {
                if (_internal_buffer && _buffer)
                {
                    delete[] static_cast<std::byte *>(_buffer);
                    _buffer = nullptr;
                }
            }

            StateType state() const noexcept
            {
                return _state;
            }

            int encode(const ValueType *data, size_t size, const WriteCallback &write_callback) noexcept
            {
                if (!data || size == 0 || size > std::numeric_limits<int>::max() || !write_callback) [[unlikely]]
                {
                    return -EINVAL;
                }
                if (!_buffer) [[unlikely]]
                {
                    return -ENOMEM;
                }

                uint8_t bytes[4];
                size_t p = 0;
                const size_t len = static_cast<size_t>(size / 3) * 3;
                for (size_t i = 0; i < len; i += 3)
                {
                    auto b0 = static_cast<uint8_t>(data[i]);
                    auto b1 = static_cast<uint8_t>(data[i + 1]);
                    auto b2 = static_cast<uint8_t>(data[i + 2]);

                    bytes[0] = _base64_chars[b0 >> 2];
                    bytes[1] = _base64_chars[((b0 & 0x03) << 4) | (b1 >> 4)];
                    bytes[2] = _base64_chars[((b1 & 0x0F) << 2) | (b2 >> 6)];
                    bytes[3] = _base64_chars[b2 & 0x3f];

                    size_t next_p = p + 4;
                    if (next_p > _buffer_size) [[unlikely]]
                    {
                        int res = write_callback(_buffer, _buffer_size);
                        if (res < 0) [[unlikely]]
                        {
                            return res;
                        }
                        p = 0;
                        next_p = 4;
                    }

                    std::memcpy(static_cast<std::byte *>(_buffer) + p, bytes, 4);
                    p = next_p;
                }

                switch (size - len)
                {
                    case 2: {
                        auto b0 = static_cast<uint8_t>(data[len]);
                        auto b1 = static_cast<uint8_t>(data[len + 1]);

                        bytes[0] = _base64_chars[b0 >> 2];
                        bytes[1] = _base64_chars[((b0 & 0x03) << 4) | (b1 >> 4)];
                        bytes[2] = _base64_chars[(b1 & 0x0F) << 2];
                        bytes[3] = '=';

                        size_t next_p = p + 4;
                        if (next_p > _buffer_size) [[unlikely]]
                        {
                            int res = write_callback(_buffer, _buffer_size);
                            if (res < 0) [[unlikely]]
                            {
                                return res;
                            }
                            p = 0;
                            next_p = 4;
                        }

                        std::memcpy(static_cast<std::byte *>(_buffer) + p, bytes, 4);
                        p = next_p;
                        break;
                    }
                    case 1: {
                        auto b0 = static_cast<uint8_t>(data[len]);

                        bytes[0] = _base64_chars[b0 >> 2];
                        bytes[1] = _base64_chars[(b0 & 0x03) << 4];
                        bytes[2] = '=';
                        bytes[3] = '=';

                        size_t next_p = p + 4;
                        if (next_p > _buffer_size) [[unlikely]]
                        {
                            int res = write_callback(_buffer, _buffer_size);
                            if (res < 0) [[unlikely]]
                            {
                                return res;
                            }
                            p = 0;
                            next_p = 4;
                        }

                        std::memcpy(static_cast<std::byte *>(_buffer) + p, bytes, 4);
                        p = next_p;
                        break;
                    }
                    default:
                        break;
                }

                if (p > 0)
                {
                    int res = write_callback(_buffer, p);
                    if (res < 0) [[unlikely]]
                    {
                        return res;
                    }
                }

                return static_cast<int>(size);
            }

        private:
            ASCII::State _state;
            void *_buffer;
            size_t _buffer_size;
            bool _internal_buffer;

            constexpr static inline const std::array<char, 64> _base64_chars = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/' };
        };

    } // namespace detail

} // namespace encoder

using EncoderASCIIBase64 = encoder::detail::ASCIIBase64;

} // namespace core

#endif
