#pragma once
#ifndef I2SXMOS_HPP
#define I2SXMOS_HPP

#include "hal/sensor.hpp"

#include <driver/gpio.h>
#include <driver/i2s_std.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace porting {

using namespace hal;

class SensorI2SXMOS final: public Sensor
{
public:
    static inline core::ConfigObjectMap DEFAULT_CONFIGS() noexcept
    {
        return { CONFIG_OBJECT_DECL_INTEGER("bclk_pin", "I2S BCLK pin", 8, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("ws_pin", "I2S WS/LRCLK pin", 7, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("dout_pin", "I2S DOUT pin (unused for RX)", 43, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("din_pin", "I2S DIN pin", 44, 0, 48),
            CONFIG_OBJECT_DECL_INTEGER("sr", "PCM sample rate in Hz", 16000, 8000, 48000),
            CONFIG_OBJECT_DECL_INTEGER("buffered_duration", "DMA buffered duration (s)", 2, 1, 5) };
    }

    SensorI2SXMOS() noexcept : Sensor(Info(3, "I2S XMOS (ReSpeaker)", Type::Microphone, { DEFAULT_CONFIGS() })) { }

    core::Status init() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status != Status::Uninitialized) [[unlikely]]
        {
            return STATUS(ENXIO, "Sensor is already initialized or in an invalid state");
        }

        const size_t sr = _info.configs["sr"].getValue<int>();
        _buffered_duration = _info.configs["buffered_duration"].getValue<int>();
        _data_buffer_capacity_frames = _buffered_duration * sr;

        if (!_data_buffer) [[likely]]
        {
            _data_buffer_capacity_bytes = _data_buffer_capacity_frames * 1 * sizeof(int32_t);
            _data_buffer = std::shared_ptr<std::byte[]>(new std::byte[_data_buffer_capacity_bytes]);
            if (!_data_buffer) [[unlikely]]
            {
                LOG(ERROR, "Failed to allocate data buffer, size: %zu bytes", _data_buffer_capacity_bytes);
                return STATUS(ENOMEM, "Failed to allocate data buffer");
            }
        }

        if (!_rx_chan) [[likely]]
        {
            _rx_chan_cfg.role = I2S_ROLE_SLAVE;
            const uint32_t dma_frame_count = 800;
            const uint32_t dma_desc_count = _data_buffer_capacity_frames / dma_frame_count;
            _rx_chan_cfg.dma_desc_num = dma_desc_count;
            _rx_chan_cfg.dma_frame_num = dma_frame_count;

            auto ret = i2s_new_channel(&_rx_chan_cfg, nullptr, &_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to create I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to create I2S channel");
            }

            ret = i2s_channel_register_event_callback(_rx_chan, &_event_cbs, this);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to register I2S channel event callback: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to register I2S channel event callback");
            }
        }

        {
            const gpio_num_t bclk = static_cast<gpio_num_t>(_info.configs["bclk_pin"].getValue<int>());
            const gpio_num_t ws = static_cast<gpio_num_t>(_info.configs["ws_pin"].getValue<int>());
            const gpio_num_t dout = static_cast<gpio_num_t>(_info.configs["dout_pin"].getValue<int>());
            const gpio_num_t din = static_cast<gpio_num_t>(_info.configs["din_pin"].getValue<int>());
            const uint32_t sample_rate = static_cast<uint32_t>(_info.configs["sr"].getValue<int>());

            _std_cfg = {
                .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
                .slot_cfg = {
                    .data_bit_width = I2S_DATA_BIT_WIDTH_32BIT,
                    .slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT,
                    .slot_mode = I2S_SLOT_MODE_MONO,
                    .slot_mask = I2S_STD_SLOT_LEFT,
                    .ws_width = 32,
                    .ws_pol = false,
                    .bit_shift = true,
                    .left_align = false,
                },
                .gpio_cfg = {
                    .mclk = I2S_GPIO_UNUSED,
                    .bclk = bclk, .ws   = ws, .dout = dout, .din  = din,
                    .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
                },
            };

            auto ret_rx = i2s_channel_init_std_mode(_rx_chan, &_std_cfg);
            if (ret_rx != ESP_OK) [[unlikely]]
                return STATUS(EIO, "Failed to init I2S STD RX mode");
        }

        _frame_index = 0;
        _data_bytes_available = 0;
        {
            auto ret_rx_en = i2s_channel_enable(_rx_chan);
            if (ret_rx_en != ESP_OK) [[unlikely]]
            {
                return STATUS(EIO, "Failed to enable I2S RX channel");
            }
        }

        _info.status = Status::Idle;

        return STATUS_OK();
    }

    core::Status deinit() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status == Status::Locked) [[unlikely]]
        {
            return STATUS(EBUSY, "Sensor is locked");
        }

        _data_buffer_capacity_frames = 0;
        _data_buffer_capacity_bytes = 0;

        _frame_index = 0;
        _data_bytes_available = 0;

        if (_data_buffer) [[likely]]
        {
            if (_data_buffer.use_count() > 1) [[unlikely]]
            {
                LOG(ERROR, "Data buffer is already in use by another process");
                return STATUS(EBUSY, "Data buffer is already in use");
            }

            _data_buffer.reset();
        }

        if (_rx_chan) [[likely]]
        {
            auto ret = i2s_channel_disable(_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to disable I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to disable I2S channel");
            }
            ret = i2s_del_channel(_rx_chan);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to delete I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to delete I2S channel");
            }
            _rx_chan = nullptr;
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    core::Status updateConfig(const core::ConfigMap &) noexcept override
    {
        return STATUS(ENOTSUP, "Update config is not supported for SensorI2SXMOS");
    }

    inline size_t dataAvailable() const noexcept override
    {
        if (!initialized()) [[unlikely]]
            return 0;
        const std::lock_guard<std::mutex> lock(_lock);

        return _data_bytes_available.load(std::memory_order_acquire) / (1 * sizeof(int32_t));
    }

    inline size_t dataClear() noexcept override
    {
        if (!initialized()) [[unlikely]]
            return 0;
        const std::lock_guard<std::mutex> lock(_lock);
        const size_t frames = _data_bytes_available.load(std::memory_order_acquire) / (1 * sizeof(int32_t));
        return internalDataDiscard(frames);
    }

    inline core::Status readDataFrame(core::DataFrame<std::unique_ptr<core::Tensor>> &data_frame,
        size_t batch_size) noexcept override
    {
        if (batch_size == 0) [[unlikely]]
        {
            LOG(ERROR, "Batch size cannot be zero");
            return STATUS(EINVAL, "Batch size cannot be zero");
        }

        if (!initialized()) [[unlikely]]
        {
            LOG(ERROR, "Sensor is not initialized");
            return STATUS(ENXIO, "Sensor is not initialized");
        }
        if (_info.status != Status::Idle) [[unlikely]]
        {
            LOG(ERROR, "Sensor is not in idle state");
            return STATUS(EINVAL, "Sensor is not in idle state");
        }
        if (_data_buffer.use_count() > 1) [[unlikely]]
        {
            LOG(ERROR, "Data buffer is already in use by another process");
            return STATUS(EBUSY, "Data buffer is already in use");
        }

        if (batch_size > _data_buffer_capacity_frames) [[unlikely]]
        {
            LOG(WARNING, "Batch size exceeds buffer capacity: %zu > %zu", batch_size, _data_buffer_capacity_frames);
            batch_size = _data_buffer_capacity_frames;
        }

        const std::lock_guard<std::mutex> lock(_lock);

        _info.status = Status::Locked;

        data_frame.timestamp = std::chrono::steady_clock::now();

        size_t read = 0;
        {
            size_t size = batch_size * 1 * sizeof(int32_t);
            auto ret = i2s_channel_read(_rx_chan, _data_buffer.get(), size, &read,
                pdMS_TO_TICKS((_buffered_duration + 1) * 1000));
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to read data from I2S channel: %s", esp_err_to_name(ret));
                _info.status = Status::Idle;
                return STATUS(EIO, "Failed to read data from I2S channel");
            }
            internalConsumeData(read);

            if (read != size) [[unlikely]]
            {
                LOG(WARNING, "Read %zu bytes, expected %zu bytes", read, size);
                if (!read) [[unlikely]]
                {
                    _info.status = Status::Idle;
                    return STATUS(EIO, "No data read from I2S channel");
                }
                batch_size = read / (1 * sizeof(int32_t));
                read = batch_size * (1 * sizeof(int32_t));
            }
        }

        auto mono_tensor
            = core::Tensor::create(core::Tensor::Type::Int16, core::Tensor::Shape(static_cast<int>(batch_size), 1));
        if (!mono_tensor)
        {
            _info.status = Status::Idle;
            return STATUS(ENOMEM, "Failed to allocate mono tensor");
        }
        auto *mono_out = mono_tensor->data<int16_t>();
        const int32_t *mono_in = reinterpret_cast<const int32_t *>(_data_buffer.get());

        for (size_t i = 0; i < batch_size; ++i)
        {
            const int32_t sample_32bit = mono_in[i];
            mono_out[i] = static_cast<int16_t>(sample_32bit >> 16);
        }

        data_frame.data = std::move(mono_tensor);
        data_frame.index = _frame_index;
        _frame_index += batch_size;

        _info.status = Status::Idle;

        return data_frame.data ? STATUS_OK() : STATUS(ENOMEM, "Failed to create data frame tensor");
    }

protected:
    static inline bool isrOnReceive(i2s_chan_handle_t, i2s_event_data_t *event, void *user_ctx)
    {
        if (!event || !user_ctx) [[unlikely]]
        {
            return false;
        }
        auto *self = static_cast<SensorI2SXMOS *>(user_ctx);

        size_t size = event->size;
        size_t data_bytes_available = self->_data_bytes_available.load(std::memory_order_acquire);
        while (1)
        {
            const size_t new_data_bytes_available
                = std::min(data_bytes_available + size, self->_data_buffer_capacity_bytes);
            if (self->_data_bytes_available.compare_exchange_strong(data_bytes_available, new_data_bytes_available,
                    std::memory_order_release, std::memory_order_relaxed)) [[likely]]
            {
                return false;
            }
        }
        return false;
    }

private:
    inline size_t internalDataDiscard(size_t size) noexcept
    {
        size_t discarded = 0;
        while (discarded < size)
        {
            size_t read
                = std::min(static_cast<size_t>((size - discarded) * 1 * sizeof(int32_t)), _data_buffer_capacity_bytes);
            auto ret = i2s_channel_read(_rx_chan, _data_buffer.get(), read, &read,
                pdMS_TO_TICKS((_buffered_duration + 1) * 1000));
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to read data from I2S channel: %s", esp_err_to_name(ret));
            }
            internalConsumeData(read);
            discarded += read;
        }
        return size;
    }

    inline void internalConsumeData(size_t size) noexcept
    {
        if (!size) [[unlikely]]
            return;
        size_t data_bytes_available = _data_bytes_available.load(std::memory_order_acquire);
        while (1)
        {
            size = std::min(size, data_bytes_available);
            const size_t new_data_bytes_available = data_bytes_available - size;
            if (_data_bytes_available.compare_exchange_strong(data_bytes_available, new_data_bytes_available,
                    std::memory_order_release, std::memory_order_relaxed)) [[likely]]
                return;
        }
    }

    mutable std::mutex _lock;

    std::shared_ptr<std::byte[]> _data_buffer = nullptr;
    size_t _data_buffer_capacity_frames = 0;
    size_t _data_buffer_capacity_bytes = 0;

    size_t _buffered_duration = 0;
    size_t _frame_index = 0;
    static inline std::atomic<size_t> _data_bytes_available = 0;

    i2s_chan_handle_t _rx_chan = nullptr;
    i2s_chan_config_t _rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_SLAVE);
    i2s_std_config_t _std_cfg = {};
    i2s_event_callbacks_t _event_cbs
        = { .on_recv = isrOnReceive, .on_recv_q_ovf = nullptr, .on_sent = nullptr, .on_send_q_ovf = nullptr };
};

} // namespace porting

#endif