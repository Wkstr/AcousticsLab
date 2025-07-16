#pragma once
#ifndef ADPCM_HPP
#define ADPCM_HPP

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <new>

#include "core/encoder.hpp"

namespace core {

namespace encoder {

    class ADPCM
    {
    public:
        using ValueType = int16_t;
        using WriteCallback = std::function<int(const void *data, size_t size)>;

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
            ADPCMIMA(void *buffer = nullptr, size_t buffer_size = 1024) noexcept
                : _state(), _buffer(buffer), _buffer_size(buffer_size), _internal_buffer(false)
            {
                if (!_buffer && _buffer_size > 0)
                {
                    _buffer = new std::byte[_buffer_size];
                    _internal_buffer = true;
                }
            }

            ~ADPCMIMA() noexcept
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

                size = (size >> 1) << 1;

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
                    _state.predictor = std::clamp(_state.predictor + predictor_diff, -32768, 32767);
                    _state.step_index = std::clamp(static_cast<int>(_state.step_index + _index_table[nibble]), 0,
                        static_cast<int>(_step_table.size() - 1));

                    if (i & 1)
                    {
                        static_cast<std::byte *>(_buffer)[p++] |= static_cast<std::byte>(nibble);
                    }
                    else
                    {
                        static_cast<std::byte *>(_buffer)[p] = static_cast<std::byte>(nibble << 4);
                    }

                    if (p >= _buffer_size) [[unlikely]]
                    {
                        int res = write_callback(_buffer, _buffer_size);
                        if (res < 0) [[unlikely]]
                        {
                            return res;
                        }
                        p = 0;
                    }
                }

                if (p > 0)
                {
                    int res = write_callback(_buffer, p);
                    if (res < 0) [[unlikely]]
                    {
                        return res;
                    }
                }

                return size;
            }

        private:
            StateType _state;
            void *_buffer;
            size_t _buffer_size;
            bool _internal_buffer;

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
