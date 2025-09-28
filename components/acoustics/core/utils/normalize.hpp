#pragma once
#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

namespace core {

class RMSNormalize1D final
{
public:
    struct Options final
    {
        float target_rms = 0.75f;
        float smoothing_factor = 0.95f;
        float fp_min = -1.0f;
        float fp_max = 1.0f;
    };

    template<typename T = std::unique_ptr<RMSNormalize1D>>
    static T create(const Options &options = Options()) noexcept
    {
        return std::make_unique<RMSNormalize1D>(options);
    }

    RMSNormalize1D(const Options &options = Options()) noexcept : _options(options), _smoothed_gain(1.0f) { }
    ~RMSNormalize1D() noexcept = default;

    template<typename T, typename P = float,
        std::enable_if_t<(std::is_integral_v<T> || std::is_floating_point_v<T>) && std::is_floating_point_v<P>, bool>
        = true>
    bool operator()(T *inout_data, size_t inout_size) noexcept
    {
        if (!inout_data || inout_size == 0) [[unlikely]]
        {
            return false;
        }

        P sum = 0.0;
        for (size_t i = 0; i < inout_size; ++i)
        {
            sum += inout_data[i] * inout_data[i];
        }

        P rms = std::sqrt(sum / inout_size);
        if (rms < std::numeric_limits<P>::epsilon()) [[unlikely]]
        {
            rms = std::numeric_limits<P>::epsilon();
        }

        P gain = _options.target_rms / rms;

        float min {}, max {};
        if constexpr (std::is_integral_v<T>)
        {
            min = std::numeric_limits<T>::min();
            max = std::numeric_limits<T>::max();
        }
        else
        {
            min = _options.fp_min;
            max = _options.fp_max;
        }

        for (size_t i = 0; i < inout_size; ++i)
        {
            _smoothed_gain = (_options.smoothing_factor * _smoothed_gain) + ((1.0f - _options.smoothing_factor) * gain);
            float normalized_sample = static_cast<float>(inout_data[i]) * _smoothed_gain;
            inout_data[i] = static_cast<T>(std::clamp(normalized_sample, min, max));
        }

        return true;
    }

private:
    Options _options;
    float _smoothed_gain;
};

} // namespace core

#endif
