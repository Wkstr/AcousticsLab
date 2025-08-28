#pragma once
#ifndef ADPCM_HPP
#define ADPCM_HPP

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

#include "core/encoder.hpp"
#include "core/logger.hpp"

namespace core {

namespace encoder {

    class ADPCM
    {
    public:
        using ValueType = int16_t;

        struct State final
        {
            State() noexcept : predictor(0), step_index(0) { }

            ~State() noexcept = default;

            ValueType predictor;
            uint8_t step_index;
        };
    };

    namespace detail {

        class ADPCMIMA final: public Encoder<ADPCM, ADPCMIMA>
        {
        public:
            template<typename T = std::unique_ptr<ADPCMIMA>>
            static T create(void *buffer = nullptr, size_t buffer_size = 4096) noexcept
            {
                bool internal_buffer = false;
                if (!buffer)
                {
                    buffer = new (std::nothrow) std::byte[buffer_size];
                    internal_buffer = true;
                    if (!buffer)
                    {
                        LOG(ERROR, "Failed to allocate internal buffer, size: %zu", buffer_size);
                        return {};
                    }
                }

                auto ptr = T { new (std::nothrow) ADPCMIMA(buffer, buffer_size, internal_buffer) };
                if (!ptr)
                {
                    LOG(ERROR, "Failed to create ADPCMIMA encoder");
                    if (internal_buffer)
                    {
                        delete[] static_cast<std::byte *>(buffer);
                    }
                    return {};
                }
                return ptr;
            }

            ~ADPCMIMA() noexcept
            {
                if (_internal_buffer && _buffer)
                {
                    delete[] static_cast<std::byte *>(_buffer);
                    _buffer = nullptr;
                }
            }

            State state() const noexcept
            {
                return _state;
            }

            template<typename T, std::enable_if_t<std::is_invocable_r_v<int, T, const void *, size_t>, bool> = true>
            size_t encode(const ValueType *data, size_t size, T &&write_callback) noexcept
            {
                if (!data || size == 0) [[unlikely]]
                {
                    _error = EINVAL;
                    return 0;
                }
                if (!_buffer) [[unlikely]]
                {
                    _error = ENOMEM;
                    return 0;
                }

                size = size - (size & 1);

                size_t p = 0;
                for (size_t i = 0; i < size; ++i)
                {
                    uint8_t nibble = 0x0;
                    int diff = data[i] - _state.predictor;
                    if (diff < 0)
                    {
                        nibble = 0x8;
                        diff = ~diff + 1;
                    }

                    int step_size = _step_table[_state.step_index];
                    int predictor_diff = step_size >> 3;

                    if (diff >= step_size)
                    {
                        nibble |= 0x4;
                        predictor_diff += step_size;
                        diff -= step_size;
                    }

                    step_size >>= 1;
                    if (diff >= step_size)
                    {
                        nibble |= 0x2;
                        predictor_diff += step_size;
                        diff -= step_size;
                    }

                    step_size >>= 1;
                    if (diff >= step_size)
                    {
                        nibble |= 0x1;
                        predictor_diff += step_size;
                    }

                    if ((nibble & 0x8) == 0x8)
                    {
                        predictor_diff = ~predictor_diff + 1;
                    }
                    _state.predictor = std::clamp(_state.predictor + predictor_diff,
                        static_cast<int>(std::numeric_limits<ValueType>::min()),
                        static_cast<int>(std::numeric_limits<ValueType>::max()));
                    _state.step_index = std::clamp(static_cast<int>(_state.step_index + _index_table[nibble]), 0,
                        static_cast<int>(_step_table.size() - 1));

                    if (i & 1)
                    {
                        static_cast<std::byte *>(_buffer)[p] |= static_cast<std::byte>(nibble);
                        if (++p >= _buffer_size) [[unlikely]]
                        {
                            int res = write_callback(_buffer, _buffer_size);
                            if (res != 0) [[unlikely]]
                            {
                                _error = res;
                                return 0;
                            }
                            p = 0;
                        }
                    }
                    else
                    {
                        static_cast<std::byte *>(_buffer)[p] = static_cast<std::byte>(nibble << 4);
                    }
                }

                if (p > 0)
                {
                    int res = write_callback(_buffer, p);
                    if (res != 0) [[unlikely]]
                    {
                        _error = res;
                        return 0;
                    }
                }

                return size;
            }

            static size_t estimate(size_t size) noexcept
            {
                return (size - (size & 1)) >> 1;
            }

        private:
            explicit ADPCMIMA(void *buffer, size_t buffer_size, bool internal_buffer) noexcept
                : Encoder(std::numeric_limits<size_t>::max() - 1), _state(), _buffer(buffer), _buffer_size(buffer_size),
                  _internal_buffer(internal_buffer)
            {
            }

            State _state;
            void *_buffer;
            const size_t _buffer_size;
            const bool _internal_buffer;

            static inline constexpr const std::array<int16_t, 89> _step_table
                = { 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80,
                      88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544,
                      598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749,
                      3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635,
                      13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767 };
            static inline constexpr const std::array<int16_t, 16> _index_table
                = { -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 };
        };

    } // namespace detail

} // namespace encoder

using EncoderADPCMIMA = core::encoder::detail::ADPCMIMA;

} // namespace core

#endif
