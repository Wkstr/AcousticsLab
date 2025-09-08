#pragma once
#ifndef SPEECH_COMMANDS_PRE_NODE_HPP
#define SPEECH_COMMANDS_PRE_NODE_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_node.hpp"

#include <dl_rfft.h>
#include <esp_heap_caps.h>

#include <cstdint>
#include <memory>

namespace porting { namespace algorithms { namespace node {

    class SpeechCommandsPreprocess final: public module::MNode
    {
    public:
        static constexpr size_t AUDIO_SAMPLES = 44032;
        static constexpr size_t FRAME_LEN = 2048;
        static constexpr size_t HOP_LEN = 1024;
        static constexpr size_t NUM_FRAMES = 43;
        static constexpr size_t FFT_SIZE = 2048;
        static constexpr size_t FEATURES_PER_FRAME = 232;
        static constexpr size_t OUTPUT_SIZE = NUM_FRAMES * FEATURES_PER_FRAME;

        static std::shared_ptr<module::MNode> create(const core::ConfigMap &configs, const module::MIOS *inputs,
            const module::MIOS *outputs, int priority)
        {
            if (inputs && inputs->size() > 1)
            {
                return nullptr;
            }
            if (outputs && outputs->size() > 1)
            {
                return nullptr;
            }

            module::MIOS input_mios = inputs ? *inputs : module::MIOS {};
            module::MIOS output_mios = outputs ? *outputs : module::MIOS {};

            if (input_mios.empty())
            {
                auto input_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Int16,
                    core::Tensor::Shape(static_cast<int>(AUDIO_SAMPLES)));
                if (!input_tensor)
                {
                    return nullptr;
                }
                auto input_mio = std::make_shared<module::MIO>(input_tensor, "audio_input");
                input_mios.push_back(input_mio);
            }

            if (output_mios.empty())
            {
                auto output_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Float32,
                    core::Tensor::Shape(static_cast<int>(OUTPUT_SIZE)));
                if (!output_tensor)
                {
                    return nullptr;
                }
                auto output_mio = std::make_shared<module::MIO>(output_tensor, "feature_data");
                output_mios.push_back(output_mio);
            }

            auto validate_status = validateTensors(input_mios, output_mios);
            if (!validate_status)
            {
                return nullptr;
            }

            auto node = std::shared_ptr<SpeechCommandsPreprocess>(
                new SpeechCommandsPreprocess(configs, std::move(input_mios), std::move(output_mios), priority));

            if (!node)
            {
                return nullptr;
            }

            auto init_status = node->initialize();
            if (!init_status)
            {
                return nullptr;
            }

            return node;
        }

        static core::Status validateTensors(const module::MIOS &inputs, const module::MIOS &outputs) noexcept
        {
            if (inputs.empty())
            {
                if (!outputs.empty() && outputs.size() != 1)
                {
                    return STATUS(EINVAL, "SpeechCommandsPreprocess requires exactly 1 output tensor when provided");
                }
                return STATUS_OK();
            }

            if (inputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommandsPreprocess requires exactly 1 input tensor");
            }

            if (outputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommandsPreprocess requires exactly 1 output tensor");
            }

            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

            if (!input_tensor || !output_tensor)
            {
                return STATUS(EINVAL, "Input or output tensor is null");
            }

            if (input_tensor->dtype() != core::Tensor::Type::Int16)
            {
                return STATUS(EINVAL, "Input tensor data type mismatch");
            }

            if (input_tensor->shape().dot() != static_cast<int>(AUDIO_SAMPLES))
            {
                return STATUS(EINVAL, "Input tensor size mismatch");
            }

            if (output_tensor->dtype() != core::Tensor::Type::Float32)
            {
                return STATUS(EINVAL, "Output tensor data type mismatch");
            }

            if (output_tensor->shape().dot() != static_cast<size_t>(OUTPUT_SIZE))
            {
                return STATUS(EINVAL, "Output tensor size mismatch");
            }

            return STATUS_OK();
        }

    private:
        inline SpeechCommandsPreprocess(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs,
            int priority)
            : module::MNode("SpeechCommandsPreprocess", std::move(inputs), std::move(outputs), priority),
              _fft_handle(nullptr), _fft_input_buffer(nullptr), _audio_float_buffer(nullptr), _blackman_window(nullptr)
        {
        }

    public:
        inline ~SpeechCommandsPreprocess() override
        {
            if (_fft_handle)
            {
                dl_rfft_f32_deinit(_fft_handle);
                _fft_handle = nullptr;
            }
        }

    private:
        inline core::Status initialize() noexcept
        {
            void *fft_ptr = heap_caps_aligned_alloc(16, FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!fft_ptr)
            {
                return STATUS(ENOMEM, "Failed to allocate FFT input buffer");
            }
            _fft_input_buffer = std::shared_ptr<float[]>(static_cast<float *>(fft_ptr), [](float *p) {
                if (p)
                    heap_caps_free(p);
            });

            void *audio_ptr
                = heap_caps_aligned_alloc(16, AUDIO_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!audio_ptr)
            {
                return STATUS(ENOMEM, "Failed to allocate audio float buffer");
            }
            _audio_float_buffer = std::shared_ptr<float[]>(static_cast<float *>(audio_ptr), [](float *p) {
                if (p)
                    heap_caps_free(p);
            });

            void *window_ptr
                = heap_caps_aligned_alloc(16, FRAME_LEN * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            if (!window_ptr)
            {
                return STATUS(ENOMEM, "Failed to allocate Blackman window buffer");
            }
            _blackman_window = std::shared_ptr<float[]>(static_cast<float *>(window_ptr), [](float *p) {
                if (p)
                    heap_caps_free(p);
            });

            _fft_handle = dl_rfft_f32_init(FFT_SIZE, MALLOC_CAP_SPIRAM);
            if (!_fft_handle)
            {
                return STATUS(EFAULT, "ESP-DL RFFT initialization failed");
            }

            for (size_t n = 0; n < FRAME_LEN; ++n)
            {
                float n_norm = static_cast<float>(n) / static_cast<float>(FRAME_LEN);
                _blackman_window[n] = 0.42f - 0.5f * std::cos(2.0f * std::numbers::pi_v<float> * n_norm)
                                      + 0.08f * std::cos(4.0f * std::numbers::pi_v<float> * n_norm);
            }

            return STATUS_OK();
        }

    protected:
        inline core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override
        {
            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

            const int16_t *raw_audio_data = input_tensor->data<int16_t>();
            float *output_features = output_tensor->data<float>();

            if (!raw_audio_data || !output_features)
            {
                return STATUS(EFAULT, "Failed to get tensor data pointers");
            }

            for (size_t i = 0; i < AUDIO_SAMPLES; ++i)
            {
                _audio_float_buffer[i] = static_cast<float>(raw_audio_data[i]) / 32768.0f;
            }

            for (size_t frame_idx = 0; frame_idx < NUM_FRAMES; ++frame_idx)
            {
                prepareFrame(_audio_float_buffer.get(), frame_idx);

                esp_err_t fft_result = dl_rfft_f32_run(_fft_handle, _fft_input_buffer.get());

                float *current_log_features = output_features + (frame_idx * FEATURES_PER_FRAME);

                for (size_t j = 0; j < FEATURES_PER_FRAME * 2; j += 2)
                {
                    float real = _fft_input_buffer[j];
                    float imag = _fft_input_buffer[j + 1];
                    float mag_sq = real * real + imag * imag;
                    current_log_features[j >> 1] = 0.5f * std::logf(mag_sq + std::numeric_limits<float>::epsilon());
                }
            }

            normalizeGlobally(output_features, OUTPUT_SIZE);

            return STATUS_OK();
        }

    private:
        dl_fft_f32_t *_fft_handle;

        std::shared_ptr<float[]> _fft_input_buffer;
        std::shared_ptr<float[]> _audio_float_buffer;
        std::shared_ptr<float[]> _blackman_window;

        inline void prepareFrame(const float *audio_data, size_t frame_idx) noexcept
        {
            size_t start_in_padded = frame_idx * HOP_LEN;
            size_t padding_size = HOP_LEN;
            if (start_in_padded < padding_size)
            {
                size_t zero_count = padding_size - start_in_padded;
                size_t copy_count = padding_size - zero_count;
                std::memset(_fft_input_buffer.get(), 0, zero_count * sizeof(float));
                std::memcpy(_fft_input_buffer.get() + zero_count, audio_data, copy_count * sizeof(float));
            }
            else
            {
                std::memcpy(_fft_input_buffer.get(), audio_data + (start_in_padded - padding_size),
                    HOP_LEN * sizeof(float));
            }

            std::memcpy(_fft_input_buffer.get() + HOP_LEN, audio_data + (frame_idx * HOP_LEN), HOP_LEN * sizeof(float));

            for (size_t i = 0; i < FRAME_LEN; ++i)
            {
                _fft_input_buffer[i] *= _blackman_window[i];
            }
        }

        inline void normalizeGlobally(float *features, size_t total_features) noexcept
        {
            float mean = 0.0;
            for (size_t i = 0; i < total_features; ++i)
            {
                mean += features[i];
            }
            mean /= total_features;

            float variance = 0.0;
            for (size_t i = 0; i < total_features; ++i)
            {
                float diff = features[i] - mean;
                variance += diff * diff;
            }
            variance /= total_features;
            float std_dev = std::sqrtf(variance);

            const float epsilon = std::numeric_limits<float>::epsilon();
            float inv_std = 1.0f / (std_dev + epsilon);

            for (size_t i = 0; i < total_features; ++i)
            {
                features[i] = (features[i] - static_cast<float>(mean)) * inv_std;
            }
        }
    };
}}} // namespace porting::algorithms::node

#endif // SPEECH_COMMANDS_PRE_NODE_HPP
