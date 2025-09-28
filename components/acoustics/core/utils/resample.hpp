#pragma once
#ifndef RESAMPLE_HPP
#define RESAMPLE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

namespace core {

template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
class ResampleLinear1D final
{
public:
    template<typename U, typename P = std::unique_ptr<ResampleLinear1D<T>>>
    static P create()
    {
        if constexpr (std::is_same_v<P, std::unique_ptr<ResampleLinear1D<T>>>)
        {
            return std::make_unique<ResampleLinear1D<T>>();
        }
        else if constexpr (std::is_same_v<P, std::shared_ptr<ResampleLinear1D<T>>>)
        {
            return std::make_shared<ResampleLinear1D<T>>();
        }
        else
        {
            return nullptr;
        }
    }

    ResampleLinear1D() noexcept = default;
    ~ResampleLinear1D() noexcept = default;

    bool operator()(const T *in_data, size_t in_size, T *out_data, size_t out_size, size_t in_rate,
        size_t out_rate) const noexcept
    {
        if (!in_data || in_size == 0 || !out_data || out_size == 0 || in_rate == 0 || out_rate == 0) [[unlikely]]
        {
            return false;
        }

        if (in_rate == out_rate) [[unlikely]]
        {
            out_size = std::min(in_size, out_size);
            std::memcpy(out_data, in_data, out_size * sizeof(T));
            return true;
        }

        const double ratio = static_cast<double>(in_rate) / out_rate;
        const size_t out_samples = std::min(static_cast<size_t>(std::floor(in_size / ratio)), out_size);
        for (size_t i = 0; i < out_samples; ++i)
        {
            float in_pos = static_cast<float>(static_cast<double>(i) * ratio);
            size_t index1 = static_cast<size_t>(std::floor(in_pos));
            size_t index2 = index1 + 1;

            float fraction = in_pos - index1;
            T sample1 = in_data[index1];
            T sample2 = index2 < in_size ? in_data[index2] : sample1;

            float interpolated_sample = std::roundf(((1.0f - fraction) * sample1) + (fraction * sample2));
            out_data[i] = static_cast<T>(std::clamp(interpolated_sample,
                static_cast<float>(std::numeric_limits<T>::min()), static_cast<float>(std::numeric_limits<T>::max())));
        }
        return true;
    }
};

} // namespace core

#endif
