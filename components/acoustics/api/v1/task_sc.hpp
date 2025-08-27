#pragma once
#ifndef TASK_SC_HPP
#define TASK_SC_HPP

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/encoder/adpcm.hpp"
#include "core/encoder/ascii.hpp"
#include "core/encoder/opus.hpp"

#include "hal/device.hpp"
#include "hal/sensor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>

namespace v1 {

namespace shared {

    inline std::atomic<bool> is_sampling = false;
    inline std::atomic<bool> is_invoking = false;

    inline constexpr const size_t buffer_size = 1024;
    inline constexpr const size_t buffer_size_mask = buffer_size - 1;
    inline std::mutex buffer_mutex;
    inline size_t buffer_head = 0;
    inline size_t buffer_tail = 0;
    inline std::vector<int16_t> buffer(buffer_size);
    inline std::chrono::steady_clock::time_point buffer_base_ts = std::chrono::steady_clock::now();

    inline constexpr const size_t sample_chunk_ms = 100;

    inline constexpr const float overlap_ratio_min = 0.f;
    inline constexpr const float overlap_ratio_max = 0.6f;
    inline std::atomic<float> overlap_ratio = 0.5f;

} // namespace shared

struct TaskSC final
{
    class Storage final: public api::Task
    {
        static inline constexpr const char *sc_storage_path = CONTEXT_PREFIX CONTEXT_VERSION "/sc_storage";

    public:
        Storage(api::Context &context, hal::Transport &transport, size_t id, std::string tag) noexcept
            : api::Task(context, transport, id, v1::defaults::task_priority), _tag(std::move(tag))
        {
        }

        ~Storage() noexcept override { }

        static inline void preInitHook()
        {
            auto device = hal::DeviceRegistry::getDevice();
            if (!device || !device->initialized()) [[unlikely]]
            {
                LOG(ERROR, "Device not registered or not initialized");
                return;
            }

            {
                float overlap_ratio = shared::overlap_ratio.load();
                size_t size = sizeof(overlap_ratio);
                auto status
                    = device->load(hal::Device::StorageType::Internal, getOverlapRatioPath(), &overlap_ratio, size);
                if (!status || size != sizeof(overlap_ratio)) [[unlikely]]
                {
                    LOG(ERROR, "Failed to load overlap_ratio or size mismatch");
                    return;
                }
                shared::overlap_ratio = std::clamp(overlap_ratio, shared::overlap_ratio_min, shared::overlap_ratio_max);
            }
        }

        inline core::Status operator()(api::Executor &executor) override
        {
            auto device = hal::DeviceRegistry::getDevice();
            if (!device || !device->initialized()) [[unlikely]]
            {
                LOG(ERROR, "Device not registered or not initialized");
                return STATUS(ENODEV, "Device not registered or not initialized");
            }

            {
                float overlap_ratio = shared::overlap_ratio.load();
                return device->store(hal::Device::StorageType::Internal, getOverlapRatioPath(), &overlap_ratio,
                    sizeof(overlap_ratio));
            }
        }

    private:
        static inline std::string getOverlapRatioPath()
        {
            return std::string(sc_storage_path) + "/overlap_ratio";
        }

        const std::string _tag;
    };

    class Sample final: public api::Task
    {
    public:
        Sample(api::Context &context, hal::Transport &transport, size_t id, hal::Sensor *sensor, std::string tag,
            const volatile size_t &external_task_id) noexcept
            : api::Task(context, transport, id, v1::defaults::task_priority), _sensor(sensor), _tag(std::move(tag)),
              _external_task_id(external_task_id), _internal_task_id(_external_task_id),
              _start_time(std::chrono::steady_clock::now()), _current_id(0), _current_id_next(0), _sr(0), _fs(0),
              _adpcm_encoder(), _opus_encoder(), _base64_encoder()
        {
            shared::is_sampling = true;

            if (!_sensor) [[likely]]
            {
                LOG(ERROR, "Sensor is null");
                return;
            }

            {
                auto it = _sensor->info().configs.find("sr");
                if (it != _sensor->info().configs.end())
                {
                    auto sr = it->second.getValue<int>();
                    if (sr <= 0)
                    {
                        LOG(ERROR, "Invalid sample rate: %d", sr);
                        return;
                    }
                    _sr = sr;
                    _fs = _sr * shared::sample_chunk_ms / 1000;
                }
            }

            {
                auto adpcm_size = core::EncoderADPCMIMA::estimate(_fs);
                auto opus_size = core::EncoderLIBOPUS::estimate(_fs, _sr);
                LOG(INFO, "Estimated ADPCM size: %zu, OPUS size: %zu", adpcm_size, opus_size);

                _adpcm_encoder = core::EncoderADPCMIMA::create(nullptr, adpcm_size);
                _opus_encoder = core::EncoderLIBOPUS::create(nullptr, opus_size);

                auto base64_size = core::EncoderASCIIBase64::estimate(std::max(adpcm_size, opus_size));
                LOG(INFO, "Estimated Base64 size: %zu", base64_size);

                _base64_encoder = core::EncoderASCIIBase64::create(nullptr, base64_size);
            }

            {
                _sensor->dataClear();
            }

            {
                std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                v1::shared::buffer_head = 0;
                v1::shared::buffer_tail = 0;
                v1::shared::buffer_base_ts = _start_time;
                _current_id = v1::shared::buffer_tail;
            }
        }

        ~Sample() noexcept override
        {
            shared::is_sampling = false;
        }

        inline core::Status operator()(api::Executor &executor) override
        {
            shared::is_sampling = true;

            if (_external_task_id != _internal_task_id) [[unlikely]]
            {
                return STATUS_OK();
            }
            if (!_sensor || !_sensor->initialized()) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Accelerometer sensor not initialized"),
                    std::chrono::steady_clock::now());
            }

            const size_t available = _sensor->dataAvailable();
            if (available < _fs)
            {
                return executor.submit(getptr(), getNextDataDelay(available));
            }

            auto status = STATUS_OK();
            void (*encoder_writer)(Sample &, core::ObjectWriter &) = nullptr;
            {
                core::DataFrame<std::unique_ptr<core::Tensor>> data_frame;
                auto status = _sensor->readDataFrame(data_frame, _fs);
                if (!status) [[unlikely]]
                {
                    return replyWithStatus(status, std::chrono::steady_clock::now());
                }
                if (!data_frame.data) [[unlikely]]
                {
                    LOG(ERROR, "Data frame is null");
                    return replyWithStatus(STATUS(EFAULT, "Data frame is null"), std::chrono::steady_clock::now());
                }
                const auto &shape = data_frame.data->shape();
                if (shape.size() != 2 || shape[0] < _fs || shape[1] != 1
                    || data_frame.data->dtype() != core::Tensor::Type::Int16) [[unlikely]]
                {
                    LOG(ERROR, "Data frame shape or type mismatch: shape=(%zu,%zu), dtype=%d", shape[0], shape[1],
                        static_cast<int>(data_frame.data->dtype()));
                    return replyWithStatus(STATUS(EINVAL, "Data frame shape or type mismatch"),
                        std::chrono::steady_clock::now());
                }

                const auto size = shape[0];
                {
                    const std::lock_guard<std::mutex> lock(v1::shared::buffer_mutex);
                    const size_t head = v1::shared::buffer_head;
                    const size_t tail = v1::shared::buffer_tail;
                    const size_t next_head = head + shape[0];
                    if (static_cast<size_t>(next_head - tail) > v1::shared::buffer_size) [[unlikely]]
                    {
                        v1::shared::buffer_tail = next_head - v1::shared::buffer_size;
                    }
                    for (size_t i = 0; i < size; ++i)
                    {
                        v1::shared::buffer[(head + i) & v1::shared::buffer_size_mask]
                            = data_frame.data->data<int16_t>()[i];
                    }
                }
                _current_id_next = _current_id + size;
                if (shared::is_invoking)
                {
                    status = encodeADPCM(data_frame.data->data<int16_t>(), size);
                    encoder_writer = writeADPCM;
                }
                else
                {
                    status = encodeOPUS(data_frame.data->data<int16_t>(), size);
                    encoder_writer = writeOPUS;
                }
            }
            status = replyWithStatus(status, std::chrono::steady_clock::now(), encoder_writer);
            if (!status) [[unlikely]]
            {
                return status;
            }

            return executor.submit(getptr(), getNextDataDelay(_sensor->dataAvailable()));
        }

    private:
        core::Status replyWithStatus(core::Status status, std::chrono::steady_clock::time_point current,
            void (*encoder_writer)(Sample &, core::ObjectWriter &) = nullptr) noexcept
        {
            auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
                std::bind(&v1::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v1::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v1::ResponseType::Stream;
            writer["name"] += "PCMData";
            writer["tag"] << _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();
            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["id"] += _current_id;
                data["ts"] += std::chrono::duration_cast<std::chrono::milliseconds>(current - _start_time).count();
                if (encoder_writer)
                {
                    encoder_writer(*this, data);
                }
            }
            _current_id = _current_id_next;

            return status;
        }

        core::Status encodeADPCM(const int16_t *data, size_t size)
        {
            if (!_adpcm_encoder || !_base64_encoder) [[unlikely]]
            {
                return STATUS(EINVAL, "ADPCM encoder or Base64 encoder not initialized");
            }

            _adpcm_state = _adpcm_encoder->state();
            _adpcm_encoder->encode(data, size, [&](const void *encoded, size_t encoded_size) {
                _base64_encoder->encode(static_cast<const uint8_t *>(encoded), encoded_size,
                    [&](const void *b64, size_t b64_size) {
                        if (b64_size == 0) [[unlikely]]
                            return EFAULT;
                        _b64_str = std::string_view(static_cast<const char *>(b64), b64_size);
                        return 0;
                    });
                return _base64_encoder->error();
            });
            auto ret = _adpcm_encoder->error();
            if (ret) [[unlikely]]
            {
                return STATUS(ret, "ADPCM encoding failed");
            }
            return STATUS_OK();
        }

        static void writeADPCM(Sample &t, core::ObjectWriter &writer)
        {
            writer["ch"] += 1;
            writer["sr"] += static_cast<int>(t._sr);
            writer["bw"] += 16;
            writer["pd"] += t._adpcm_state.predictor;
            writer["si"] += t._adpcm_state.step_index;
            writer["adpcm"] << t._b64_str;
        }

        core::Status encodeOPUS(const int16_t *data, size_t size)
        {
            if (!_opus_encoder || !_base64_encoder) [[unlikely]]
            {
                return STATUS(EINVAL, "OPUS encoder or Base64 encoder not initialized");
            }

            _opus_state = _opus_encoder->state();
            _opus_encoder->encode(data, size, [&](const void *encoded, size_t encoded_size) {
                _base64_encoder->encode(static_cast<const uint8_t *>(encoded), encoded_size,
                    [&](const void *b64, size_t b64_size) {
                        if (b64_size == 0) [[unlikely]]
                            return EFAULT;
                        _b64_str = std::string_view(static_cast<const char *>(b64), b64_size);
                        return 0;
                    });
                return _base64_encoder->error();
            });
            auto ret = _opus_encoder->error();
            if (ret) [[unlikely]]
            {
                return STATUS(ret, "OPUS encoding failed");
            }
            return STATUS_OK();
        }

        static void writeOPUS(Sample &t, core::ObjectWriter &writer)
        {
            writer["ch"] += 1;
            writer["sr"] += static_cast<int>(t._opus_state.sample_rate);
            writer["bw"] += 16;
            writer["opus"] << t._b64_str;
        }

        inline size_t getNextDataDelay(size_t current_available) const noexcept
        {
            if (current_available >= _fs) [[unlikely]]
            {
                return 0;
            }
            return (_fs - current_available) * 1000 / _sr;
        }

        hal::Sensor *_sensor;
        const std::string _tag;
        const volatile size_t &_external_task_id;
        const size_t _internal_task_id;
        const std::chrono::steady_clock::time_point _start_time;
        size_t _current_id;
        size_t _current_id_next;
        size_t _sr;
        size_t _fs;

        std::unique_ptr<core::EncoderADPCMIMA> _adpcm_encoder;
        std::unique_ptr<core::EncoderLIBOPUS> _opus_encoder;
        std::unique_ptr<core::EncoderASCIIBase64> _base64_encoder;

        core::EncoderADPCMIMA::State _adpcm_state;
        core::EncoderLIBOPUS::State _opus_state;
        std::string_view _b64_str;
    };
};

} // namespace v1

#endif
