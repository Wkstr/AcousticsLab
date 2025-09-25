#pragma once
#if defined(LIB_OPUS_ENABLE) && LIB_OPUS_ENABLE

#ifndef OPUS_HPP
#define OPUS_HPP

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

#include "core/encoder.hpp"
#include "core/logger.hpp"
#include "core/utils/sound_resample.hpp"

#include <opus.h>

namespace core {

namespace encoder {

    class OPUS
    {
    public:
        using ValueType = int16_t;

        struct State final
        {
            State() noexcept : sample_rate(0) { }

            ~State() noexcept = default;

            size_t sample_rate;
        };
    };

    namespace detail {

        class LIBOPUS final: public Encoder<OPUS, LIBOPUS>
        {
        public:
            template<typename T = std::unique_ptr<LIBOPUS>>
            static T create(void *buffer = nullptr, size_t buffer_size = 4096, size_t sample_rate = 44100,
                size_t channels = 1, int application = OPUS_APPLICATION_AUDIO, size_t bitrate = 256000,
                int complexity = 9, size_t frame_ms = 100) noexcept
            {
                size_t bytes_per_second = bitrate / 8;
                size_t buffer_size_min = bytes_per_second * frame_ms / 1000;
                size_t buffer_size_target = 0;
                for (size_t i = 1; i < std::numeric_limits<int>::max() / 2; i <<= 1)
                {
                    if (i >= static_cast<size_t>(buffer_size_min))
                    {
                        buffer_size_target = i;
                        break;
                    }
                }
                if (buffer_size < buffer_size_target)
                {
                    LOG(ERROR, "Provided buffer size %zu is too small, need at least %zu", buffer_size,
                        buffer_size_target);
                    return {};
                }

                size_t frame_size = (sample_rate * frame_ms) / 1000;
                size_t sample_rate_encoder = 0;
                if (sample_rate <= 8000)
                    sample_rate_encoder = 8000;
                else if (sample_rate <= 12000)
                    sample_rate_encoder = 12000;
                else if (sample_rate <= 16000)
                    sample_rate_encoder = 16000;
                else if (sample_rate <= 24000)
                    sample_rate_encoder = 24000;
                else if (sample_rate <= 48000)
                    sample_rate_encoder = 48000;
                else
                {
                    LOG(ERROR, "Unsupported sample rate: %zu", sample_rate);
                    return {};
                }

                int err = OPUS_OK;
                auto *opus_encoder = opus_encoder_create(sample_rate_encoder, channels, application, &err);
                if (err != OPUS_OK || !opus_encoder)
                {
                    LOG(ERROR, "Failed to create OPUS encoder: %s", opus_strerror(err));
                    return {};
                }

                int frame_duration = 0;
                if (frame_ms == 5)
                    frame_duration = OPUS_FRAMESIZE_5_MS;
                else if (frame_ms == 10)
                    frame_duration = OPUS_FRAMESIZE_10_MS;
                else if (frame_ms == 20)
                    frame_duration = OPUS_FRAMESIZE_20_MS;
                else if (frame_ms == 40)
                    frame_duration = OPUS_FRAMESIZE_40_MS;
                else if (frame_ms == 60)
                    frame_duration = OPUS_FRAMESIZE_60_MS;
                else if (frame_ms == 80)
                    frame_duration = OPUS_FRAMESIZE_80_MS;
                else if (frame_ms == 100)
                    frame_duration = OPUS_FRAMESIZE_100_MS;
                else
                {
                    LOG(ERROR, "Unsupported frame duration: %d ms", frame_ms);
                    opus_encoder_destroy(opus_encoder);
                    return {};
                }

                err = opus_encoder_ctl(opus_encoder, OPUS_SET_VBR(1));
                if (err == OPUS_OK)
                    err = opus_encoder_ctl(opus_encoder, OPUS_SET_BITRATE(bitrate));
                if (err == OPUS_OK)
                    err = opus_encoder_ctl(opus_encoder, OPUS_SET_COMPLEXITY(complexity));
                if (err == OPUS_OK)
                    err = opus_encoder_ctl(opus_encoder, OPUS_SET_EXPERT_FRAME_DURATION(frame_duration));
                if (err != OPUS_OK)
                {
                    LOG(ERROR, "Failed to set OPUS encoder parameters: %s", opus_strerror(err));
                    opus_encoder_destroy(opus_encoder);
                    return {};
                }

                void *resample_buffer = nullptr;
                const size_t resample_buffer_size = ((sample_rate_encoder * frame_ms) / 1000) * sizeof(ValueType);
                if (sample_rate != sample_rate_encoder)
                {
                    if (channels != 1)
                    {
                        LOG(ERROR, "Resampling is only supported for mono audio");
                        opus_encoder_destroy(opus_encoder);
                        return {};
                    }
                    resample_buffer = new (std::nothrow) std::byte[resample_buffer_size];
                    if (!resample_buffer)
                    {
                        LOG(ERROR, "Failed to allocate resample buffer, size: %zu", resample_buffer_size);
                        opus_encoder_destroy(opus_encoder);
                        return {};
                    }
                }

                bool internal_buffer = false;
                if (!buffer)
                {
                    buffer = new (std::nothrow) std::byte[buffer_size];
                    if (!buffer)
                    {
                        LOG(ERROR, "Failed to allocate internal buffer, size: %zu", buffer_size);
                        if (resample_buffer)
                        {
                            delete[] static_cast<std::byte *>(resample_buffer);
                            resample_buffer = nullptr;
                        }
                        opus_encoder_destroy(opus_encoder);
                        return {};
                    }
                    internal_buffer = true;
                }

                auto ptr = T { new (std::nothrow) LIBOPUS(frame_size, opus_encoder, resample_buffer,
                    resample_buffer_size, sample_rate, sample_rate_encoder, buffer, buffer_size, internal_buffer) };
                if (!ptr)
                {
                    LOG(ERROR, "Failed to allocate LIBOPUS instance");
                    if (internal_buffer)
                    {
                        delete[] static_cast<std::byte *>(buffer);
                        buffer = nullptr;
                    }
                    if (resample_buffer)
                    {
                        delete[] static_cast<std::byte *>(resample_buffer);
                        resample_buffer = nullptr;
                    }
                    opus_encoder_destroy(opus_encoder);
                    return {};
                }
                return ptr;
            }

            ~LIBOPUS() noexcept
            {
                if (_internal_buffer && _buffer)
                {
                    delete[] static_cast<std::byte *>(_buffer);
                    _buffer = nullptr;
                }

                if (_resample_buffer)
                {
                    delete[] static_cast<ValueType *>(_resample_buffer);
                    _resample_buffer = nullptr;
                }

                if (_opus_encoder)
                {
                    opus_encoder_destroy(_opus_encoder);
                    _opus_encoder = nullptr;
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

                size_t encoded = 0;
                static_assert(sizeof(std::byte) == sizeof(unsigned char));
                if (_resample_buffer)
                {
                    while (size >= _frame_size)
                    {
                        if (!soundResampleLinear(data, _frame_size, static_cast<int16_t *>(_resample_buffer),
                                _resample_frame_size, _sample_rate, _sample_rate_encoder))
                        {
                            LOG(ERROR, "Failed to resample audio");
                            _error = EINVAL;
                            return encoded;
                        }
                        int res = opus_encode(_opus_encoder, static_cast<const opus_int16 *>(_resample_buffer),
                            _resample_frame_size, static_cast<unsigned char *>(_buffer), _buffer_size);
                        if (res < 0)
                        {
                            LOG(ERROR, "Failed to encode OPUS packet: %s", opus_strerror(res));
                            _error = res;
                            return encoded;
                        }
                        write_callback(_buffer, static_cast<size_t>(res));
                        encoded += _frame_size;
                        size -= _frame_size;
                    }
                }
                else
                {
                    while (size >= _frame_size) [[unlikely]]
                    {
                        int res = opus_encode(_opus_encoder, static_cast<const opus_int16 *>(data), _frame_size,
                            static_cast<unsigned char *>(_buffer), _buffer_size);
                        if (res < 0)
                        {
                            LOG(ERROR, "Failed to encode OPUS packet: %s", opus_strerror(res));
                            _error = res;
                            return encoded;
                        }
                        write_callback(_buffer, static_cast<size_t>(res));
                        encoded += _frame_size;
                        size -= _frame_size;
                    }
                }

                return encoded;
            }

            static size_t estimate(size_t size, size_t sample_rate, size_t bitrate = 256000) noexcept
            {
                if (sample_rate == 0) [[unlikely]]
                {
                    return 0;
                }
                const size_t samples
                    = static_cast<size_t>(std::ceil(static_cast<float>(size * (bitrate >> 3)) / sample_rate));
                for (size_t i = 1; i < std::numeric_limits<int>::max(); i <<= 1)
                {
                    if (i >= samples)
                    {
                        return i;
                    }
                }
                return std::numeric_limits<size_t>::max();
            }

        private:
            explicit LIBOPUS(size_t frame_size, OpusEncoder *opus_encoder, void *resample_buffer,
                size_t resample_buffer_size, size_t sample_rate, size_t sample_rate_encoder, void *buffer,
                size_t buffer_size, bool internal_buffer) noexcept
                : Encoder((std::numeric_limits<size_t>::max() >> 1) - 1), _state(), _frame_size(frame_size),
                  _resample_frame_size(resample_buffer_size / sizeof(ValueType)), _opus_encoder(opus_encoder),
                  _resample_buffer(resample_buffer), _resample_buffer_size(resample_buffer_size),
                  _sample_rate(sample_rate), _sample_rate_encoder(sample_rate_encoder), _buffer(buffer),
                  _buffer_size(buffer_size), _internal_buffer(internal_buffer)
            {
                _state.sample_rate = sample_rate_encoder;
                LOG(INFO,
                    "LIBOPUS initialized: frame_size=%zu, resample_frame_size=%zu, resample_buffer_size=%zu, "
                    "sample_rate=%zu, "
                    "sample_rate_encoder=%zu, "
                    "buffer_size=%zu",
                    _frame_size, _resample_frame_size, _resample_buffer_size, _sample_rate, _sample_rate_encoder,
                    _buffer_size);
            }

            State _state;
            const size_t _frame_size;
            const size_t _resample_frame_size;
            OpusEncoder *_opus_encoder;
            void *_resample_buffer;
            const size_t _resample_buffer_size;
            const size_t _sample_rate;
            const size_t _sample_rate_encoder;
            void *_buffer;
            const size_t _buffer_size;
            const bool _internal_buffer;
        };

    } // namespace detail

} // namespace encoder

using EncoderLIBOPUS = core::encoder::detail::LIBOPUS;

} // namespace core

#endif // OPUS_HPP

#endif
