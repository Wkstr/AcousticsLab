#pragma once
#ifndef SPEECH_COMMANDS_PREPROCESS_HPP
#define SPEECH_COMMANDS_PREPROCESS_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include "module/module_node.hpp"

#include <dl_rfft.h>
#include <esp_err.h>
#include <esp_heap_caps.h>

#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <string_view>

namespace porting::algorithms::node {

class SpeechCommandsPreprocess final: public module::MNode
{
    static constexpr inline const size_t AUDIO_SAMPLES = 44032;
    static constexpr inline const size_t FRAME_LEN = 2048;
    static constexpr inline const size_t HOP_LEN = 1024;
    static constexpr inline const size_t NUM_FRAMES = 43;
    static constexpr inline const size_t FFT_SIZE = 2048;
    static constexpr inline const size_t FEATURES_PER_FRAME = 232;
    static constexpr inline const float NORM_BOUND_I16 = -static_cast<float>(std::numeric_limits<int16_t>::min());

public:
    static inline constexpr const std::string_view node_name = "SpeechCommandsPreprocess";

    static std::shared_ptr<module::MNode> create(const core::ConfigMap &configs, const module::MIOS *inputs,
        const module::MIOS *outputs, int priority)
    {
        auto input_mios = getInputMIOS(inputs);
        if (input_mios.empty())
        {
            LOG(ERROR, "Failed to get or create input MIOS");
            return {};
        }

        auto output_mios = getOutputMIOS(outputs);
        if (output_mios.empty())
        {
            LOG(ERROR, "Failed to get or create output MIOS");
            return {};
        }

        dl_fft_f32_t *fft_handle = nullptr;
        float *fft_buffer = nullptr;
        float *frame_buffer = nullptr;
        float *window_buffer = nullptr;
        if (!(fft_handle = dl_rfft_f32_init(FFT_SIZE, MALLOC_CAP_SPIRAM))
            || !(fft_buffer = static_cast<float *>(
                     heap_caps_aligned_alloc(16, FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)))
            || !(frame_buffer = new (std::nothrow) float[HOP_LEN])
            || !(window_buffer = new (std::nothrow) float[FRAME_LEN]))
            goto Err;

        fillBlackmanWindow(window_buffer, FRAME_LEN);

        {
            std::shared_ptr<module::MNode> node(
                new (std::nothrow) SpeechCommandsPreprocess(configs, std::move(input_mios), std::move(output_mios),
                    priority, fft_handle, fft_buffer, frame_buffer, window_buffer));
            if (!node)
                goto Err;
            return node;
        }

    Err:
        LOG(ERROR, "Failed to allocate resources");
        if (fft_handle)
            dl_rfft_f32_deinit(fft_handle);
        if (fft_buffer)
            heap_caps_free(fft_buffer);
        if (frame_buffer)
            delete[] frame_buffer;
        if (window_buffer)
            delete[] window_buffer;
        return {};
    }

    inline ~SpeechCommandsPreprocess() noexcept override
    {
        if (_fft_handle)
        {
            dl_rfft_f32_deinit(_fft_handle);
            _fft_handle = nullptr;
        }
        if (_fft_buffer)
        {
            heap_caps_free(_fft_buffer);
            _fft_buffer = nullptr;
        }
        if (_frame_buffer)
        {
            delete[] _frame_buffer;
            _frame_buffer = nullptr;
        }
        if (_window_buffer)
        {
            delete[] _window_buffer;
            _window_buffer = nullptr;
        }
    }

private:
    static module::MIOS getInputMIOS(const module::MIOS *inputs) noexcept
    {
        if (inputs)
        {
            if (inputs->size() != 1)
                return {};
            const auto &mio = (*inputs)[0];
            if (!mio)
                return {};
            auto *tensor = mio->operator()();
            if (!tensor || !tensor->data() || tensor->dtype() != core::Tensor::Type::Int16
                || tensor->shape().size() != 2 || tensor->shape()[0] != AUDIO_SAMPLES || tensor->shape()[1] != 1)
                return {};
            return *inputs;
        }

        auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Int16,
            core::Tensor::Shape(static_cast<int>(AUDIO_SAMPLES), 1));
        if (!tensor)
            return {};
        auto mio = std::make_shared<module::MIO>(tensor, "pcm_input");
        if (!mio)
            return {};
        return { mio };
    }

    static module::MIOS getOutputMIOS(const module::MIOS *outputs) noexcept
    {
        if (outputs)
        {
            if (outputs->size() != 1)
                return {};
            const auto &mio = (*outputs)[0];
            if (!mio)
                return {};
            auto *tensor = mio->operator()();
            if (!tensor || !tensor->data() || tensor->dtype() != core::Tensor::Type::Float32
                || tensor->shape().size() != 4 || tensor->shape()[0] != 1 || tensor->shape()[1] != NUM_FRAMES
                || tensor->shape()[2] != FEATURES_PER_FRAME || tensor->shape()[3] != 1)
                return {};
            return *outputs;
        }

        auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Float32,
            core::Tensor::Shape(1, static_cast<int>(NUM_FRAMES), static_cast<int>(FEATURES_PER_FRAME), 1));
        if (!tensor)
            return {};
        auto mio = std::make_shared<module::MIO>(tensor, "feature_output");
        if (!mio)
            return {};
        return { mio };
    }

    static void fillBlackmanWindow(float *buffer, size_t size) noexcept
    {
        for (size_t i = 0; i < size; ++i)
        {
            float n = static_cast<float>(i) / static_cast<float>(size);
            buffer[i] = 0.42f - 0.5f * std::cos(2.0f * std::numbers::pi_v<float> * n)
                        + 0.08f * std::cos(4.0f * std::numbers::pi_v<float> * n);
        }
    }

    SpeechCommandsPreprocess(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs, int priority,
        dl_fft_f32_t *fft_handle, float *fft_buffer, float *frame_buffer, float *window_buffer) noexcept
        : module::MNode(std::string(node_name), std::move(inputs), std::move(outputs), priority),
          _fft_handle(fft_handle), _fft_buffer(fft_buffer), _frame_buffer(frame_buffer), _window_buffer(window_buffer)
    {
    }

    inline core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override
    {
        const auto &input_tensor = inputs[0]->operator()();
        const auto &output_tensor = outputs[0]->operator()();
        const int16_t *input_data = input_tensor->data<int16_t>();
        float *output_features = output_tensor->data<float>();
        if (!input_data || !output_features)
        {
            return STATUS(EFAULT, "Tensor data is null");
        }

        for (size_t frame_idx = 0; frame_idx < NUM_FRAMES; ++frame_idx)
        {
            prepareFrame(input_data, frame_idx);

            esp_err_t ret = dl_rfft_f32_run(_fft_handle, _fft_buffer);
            if (ret != ESP_OK)
            {
                return STATUS(EFAULT, std::string("FFT computation failed: ") + esp_err_to_name(ret));
            }

            float *features = output_features + (frame_idx * FEATURES_PER_FRAME);
            for (size_t i = 0; i < FEATURES_PER_FRAME * 2; i += 2)
            {
                float real = _fft_buffer[i];
                float imag = _fft_buffer[i + 1];
                features[i >> 1] = 0.5f * std::logf(real * real + imag * imag + std::numeric_limits<float>::epsilon());
            }
        }

        normalizeGlobally(output_features, output_tensor->shape().dot() > 0 ? output_tensor->shape().dot() : 0);

        return STATUS_OK();
    }

    inline void fillFrameBuffer(const int16_t *input, size_t size) noexcept
    {
        for (size_t i = 0; i < size; ++i)
        {
            _frame_buffer[i] = static_cast<float>(input[i]) / NORM_BOUND_I16;
        }
    }

    inline void prepareFrame(const int16_t *input, size_t frame_idx) noexcept
    {
        const size_t hop_bytes = HOP_LEN * sizeof(float);

        if (frame_idx != 0)
            std::memcpy(_fft_buffer, _frame_buffer, hop_bytes);
        else
            std::memset(_fft_buffer, 0, hop_bytes);

        fillFrameBuffer(input + (frame_idx * HOP_LEN), HOP_LEN);
        std::memcpy(_fft_buffer + HOP_LEN, _frame_buffer, hop_bytes);

        for (size_t i = 0; i < FRAME_LEN; ++i)
        {
            _fft_buffer[i] *= _window_buffer[i];
        }
    }

    inline void normalizeGlobally(float *features, size_t size) noexcept
    {
        float mean = 0.f;
        for (size_t i = 0; i < size; ++i)
        {
            mean += features[i];
        }
        mean /= size;

        float variance = 0.f;
        for (size_t i = 0; i < size; ++i)
        {
            float diff = features[i] - mean;
            variance += diff * diff;
        }
        variance /= size;

        float std_dev = std::sqrtf(variance);
        float inv_std = 1.f / (std_dev + std::numeric_limits<float>::epsilon());
        for (size_t i = 0; i < size; ++i)
        {
            features[i] = (features[i] - mean) * inv_std;
        }
    }

private:
    dl_fft_f32_t *_fft_handle;
    float *_fft_buffer;
    float *_frame_buffer;
    float *_window_buffer;
};

} // namespace porting::algorithms::node

#endif
