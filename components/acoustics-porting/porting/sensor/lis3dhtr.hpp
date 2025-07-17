#pragma once
#ifndef LIS3DHTR_HPP
#define LIS3DHTR_HPP

#include "core/ring_buffer.hpp"
#include "hal/sensor.hpp"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>
#include <lis3dhtr.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace porting {

using namespace hal;

class SensorLIS3DHTR final: public Sensor
{
public:
    static core::ConfigObjectMap DEFAULT_CONFIGS()
    {
        core::ConfigObjectMap configs;

        configs.emplace("sda", core::ConfigObject::createInteger("sda", "SDA pin number", 5, 1, 20));
        configs.emplace("scl", core::ConfigObject::createInteger("scl", "SCL pin number", 6, 1, 20));
        configs.emplace("fsr",
            core::ConfigObject::createInteger("fsr", "Full scale range in G (2, 4, 8, 16)", 2, 2, 16));
        configs.emplace("ord", core::ConfigObject::createInteger("ord",
                                   "Output data rate in HZ (1, 10, 25, 50, 100, 200, 400)", 200, 1, 400));
        configs.emplace("sr", core::ConfigObject::createInteger("sr", "Sample rate in Hz", 100, 1, 200));

        return configs;
    }

    SensorLIS3DHTR() : Sensor(Info(1, "LIS3DHTR Accelerometer", Type::Accelerometer, { DEFAULT_CONFIGS() })) { }

    core::Status init() override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status != Status::Uninitialized)
        {
            return STATUS(ENXIO, "Sensor is already initialized or in an invalid state");
        }

        {
            int sda = _info.configs["sda"].getValue<int>();
            int scl = _info.configs["scl"].getValue<int>();
            if (!_lis3dhtr)
            {
                _lis3dhtr = new driver::LIS3DHTR();
            }
            auto ret = _lis3dhtr->init(sda, scl);
            if (ret != 0 && ret != EALREADY)
            {
                LOG(ERROR, "Failed to initialize LIS3DHTR accelerometer: %s", std::strerror(ret));
                delete _lis3dhtr;
                _lis3dhtr = nullptr;
                return STATUS_CODE(ret);
            }
        }

        {
            int fsr = _info.configs["fsr"].getValue<int>();
            switch (fsr)
            {
                case 2:
                    _lis3dhtr->setFullScaleRange(driver::lis3dhtr_scale_type_t::LIS3DHTR_RANGE_2G);
                    break;
                case 4:
                    _lis3dhtr->setFullScaleRange(driver::lis3dhtr_scale_type_t::LIS3DHTR_RANGE_4G);
                    break;
                case 8:
                    _lis3dhtr->setFullScaleRange(driver::lis3dhtr_scale_type_t::LIS3DHTR_RANGE_8G);
                    break;
                case 16:
                    _lis3dhtr->setFullScaleRange(driver::lis3dhtr_scale_type_t::LIS3DHTR_RANGE_16G);
                    break;
                default:
                    return STATUS(EINVAL, "Invalid full scale range value");
            }
        }

        {
            int ord = _info.configs["ord"].getValue<int>();
            switch (ord)
            {
                case 1:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_1HZ);
                    break;
                case 10:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_10HZ);
                    break;
                case 25:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_25HZ);
                    break;
                case 50:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_50HZ);
                    break;
                case 100:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_100HZ);
                    break;
                case 200:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_200HZ);
                    break;
                case 400:
                    _lis3dhtr->setOutputDataRate(driver::lis3dhtr_odr_type_t::LIS3DHTR_DATARATE_400HZ);
                    break;
                default:
                    return STATUS(EINVAL, "Invalid output data rate value");
            }
        }

        {
            const int sr = _info.configs["sr"].getValue<int>();

            if (!_buffer)
            {
                static_assert(_buffer_capacity >= 3, "Buffer capacity must be at least 3");
                if (_buffer_capacity < sr * 3)
                {
                    LOG(WARNING, "Buffer capacity (%zu) is less than required for sample rate (%d) * 3",
                        _buffer_capacity, sr);
                }
                _buffer = core::RingBuffer<Data>::create(_buffer_capacity, nullptr, 0);
                if (!_buffer)
                {
                    LOG(ERROR, "Failed to allocate buffer, size: %zu bytes", _buffer_capacity);
                    return STATUS(ENOMEM, "Failed to allocate buffer");
                }
                LOG(DEBUG, "Allocated buffer of size: %zu bytes at %p", _buffer_capacity, _buffer.get());
            }

            const TickType_t period = pdMS_TO_TICKS(1000 / sr);
            if (_timer)
            {
                auto ret = xTimerDelete(_timer, portMAX_DELAY);
                if (ret != pdPASS)
                {
                    LOG(ERROR, "Failed to delete existing timer at: %p", _timer);
                    return STATUS(EFAULT, "Failed to delete existing timer");
                }
            }
            _timer = xTimerCreate("lis3dhtr", period, pdTRUE, nullptr, timerCallback);
            if (_timer == nullptr)
            {
                LOG(ERROR, "Failed to create timer");
                return STATUS(EFAULT, "Failed to create timer");
            }
            _this = this;
            auto ret = xTimerStart(_timer, portMAX_DELAY);
            if (ret != pdPASS)
            {
                LOG(ERROR, "Failed to start timer: %d", ret);
                return STATUS(EFAULT, "Failed to start timer");
            }
            _info.status = Status::Idle;
        }

        {
            _frame_index = 0;
            if (!_data_buffer)
            {
                static_assert(_data_buffer_capacity >= 3, "Data buffer capacity must be at least 3");
                _data_buffer = std::make_shared<std::byte[]>(_data_buffer_capacity * sizeof(Data));
                if (!_data_buffer)
                {
                    LOG(ERROR, "Failed to allocate data buffer of size: %zu bytes",
                        _data_buffer_capacity * sizeof(Data));
                    return STATUS(ENOMEM, "Failed to allocate data buffer");
                }
                LOG(DEBUG, "Allocated data buffer of size: %zu bytes at %p", _data_buffer_capacity * sizeof(Data),
                    _data_buffer.get());
            }
        }

        {
            _sensor_status = 0;
        }

        return STATUS_OK();
    }

    core::Status deinit() override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status <= Status::Uninitialized)
        {
            return STATUS_OK();
        }

        if (_info.status == Status::Locked)
        {
            return STATUS(EBUSY, "Sensor is locked");
        }

        _frame_index = 0;
        if (_data_buffer)
        {
            _data_buffer.reset();
        }

        if (_timer)
        {
            auto ret = xTimerStop(_timer, portMAX_DELAY);
            if (ret != pdPASS)
            {
                LOG(ERROR, "Failed to stop timer: %d", ret);
                return STATUS(EFAULT, "Failed to stop timer");
            }
            ret = xTimerDelete(_timer, portMAX_DELAY);
            if (ret != pdPASS)
            {
                LOG(ERROR, "Failed to delete timer at: %p", _timer);
                return STATUS(EFAULT, "Failed to delete timer");
            }
            _timer = nullptr;
        }

        if (_this == this)
        {
            _this = nullptr;
        }

        if (_buffer)
        {
            _buffer.reset();
        }

        if (_lis3dhtr)
        {
            _lis3dhtr->deinit();
            delete _lis3dhtr;
            _lis3dhtr = nullptr;
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    core::Status updateConfig(const core::ConfigMap &configs) override
    {
        return STATUS(ENOTSUP, "Update config is not supported for LIS3DHTR sensor");
    }

    inline size_t dataAvailable() const noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!_buffer) [[unlikely]]
        {
            LOG(ERROR, "Buffer is not initialized");
            return 0;
        }
        return _buffer->size() > _data_buffer_capacity ? _data_buffer_capacity : _buffer->size();
    }

    inline size_t dataClear() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!_buffer) [[unlikely]]
        {
            LOG(ERROR, "Buffer is not initialized");
            return 0;
        }
        const auto available = _buffer->size();
        _buffer->clear();
        return available;
    }

    inline core::Status readDataFrame(core::DataFrame<std::shared_ptr<core::Tensor>> &data_frame,
        size_t batch_size) override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!initialized()) [[unlikely]]
        {
            LOG(ERROR, "Sensor is not initialized");
            return STATUS(ENXIO, "Sensor is not initialized");
        }

        if (_sensor_status != 0) [[unlikely]]
        {
            LOG(ERROR, "Sensor status is invalid: %d", _sensor_status);
            return STATUS(_sensor_status, "Sensor status is invalid");
        }

        if (!_buffer) [[unlikely]]
        {
            LOG(ERROR, "Buffer is not initialized");
            return STATUS(EFAULT, "Buffer is not initialized");
        }

        if (batch_size > _buffer_capacity) [[unlikely]]
        {
            LOG(ERROR, "Batch size exceeds buffer capacity: %zu > %zu", batch_size, _buffer_capacity);
            return STATUS(EOVERFLOW, "Batch size exceeds buffer capacity");
        }

        if (_buffer->size() < batch_size)
        {
            LOG(DEBUG, "Not enough data in buffer: %zu < %zu", _buffer->size(), batch_size);
            return STATUS(EAGAIN, "Not enough data in buffer");
        }

        if (!_data_buffer) [[unlikely]]
        {
            LOG(ERROR, "Data buffer is not initialized");
            return STATUS(EFAULT, "Data buffer is not initialized");
        }

        if (batch_size > _data_buffer_capacity) [[unlikely]]
        {
            LOG(ERROR, "Batch size exceeds data buffer capacity: %zu > %zu", batch_size, _data_buffer_capacity);
            return STATUS(EOVERFLOW, "Batch size exceeds data buffer capacity");
        }

        _info.status = Status::Locked;

        const size_t n_elems_to_discard = _buffer->size() - batch_size;
        if (n_elems_to_discard > 0)
        {
            const auto discarded = _buffer->read(nullptr, n_elems_to_discard);
            LOG(DEBUG, "Discarded %zu staled elements from buffer", discarded);
        }
        batch_size = _buffer->read(reinterpret_cast<Data *>(_data_buffer.get()), batch_size);

        data_frame.timestamp = std::chrono::steady_clock::now();
        data_frame.index = _frame_index;
        data_frame.data = core::Tensor::create(core::Tensor::Type::Float32, core::Tensor::Shape(batch_size, 3),
            reinterpret_cast<void *>(_data_buffer.get()), _data_buffer_capacity * sizeof(Data));

        _frame_index += batch_size;

        _info.status = Status::Idle;

        return STATUS_OK();
    }

public:
    struct __attribute__((packed)) Data
    {
        float x;
        float y;
        float z;
    };

protected:
    static void timerCallback(TimerHandle_t xTimer) noexcept
    {
        if (!_this || !_this->_lis3dhtr || !_this->_buffer)
        {
            LOG(ERROR, "Timer callback called with invalid state");
            return;
        }

        float x, y, z;
        int ret = _this->_lis3dhtr->getAcceleration(x, y, z);
        if (ret != 0) [[unlikely]]
        {
            LOG(ERROR, "Failed to read acceleration data: %d", ret);
            _this->_sensor_status = ret;
            return;
        }

        Data data = { x, y, z };
        if (!_this->_buffer->put(data)) [[unlikely]]
        {
            LOG(ERROR, "Failed to push data to buffer");
            _this->_sensor_status = ENOMEM;
        }
    }

private:
    mutable std::mutex _lock;
    driver::LIS3DHTR *_lis3dhtr = nullptr;
    static constexpr size_t _buffer_capacity = 2048;
    std::unique_ptr<core::RingBuffer<Data>> _buffer;
    volatile int _sensor_status = 0;
    TimerHandle_t _timer = nullptr;
    static SensorLIS3DHTR *_this;

    size_t _frame_index = 0;
    std::shared_ptr<std::byte[]> _data_buffer = nullptr;
    static constexpr size_t _data_buffer_capacity = _buffer_capacity / 2;
};

SensorLIS3DHTR *SensorLIS3DHTR::_this = nullptr;

} // namespace porting

#endif
