#pragma once
#ifndef UART_CONSOLE_HPP
#define UART_CONSOLE_HPP

#include "core/ring_buffer.hpp"
#include "hal/transport.hpp"

#include <driver/usb_serial_jtag.h>
#include <esp_err.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>

namespace porting {

class TransportUARTConsole final: public hal::Transport
{
public:
    static inline core::ConfigObjectMap DEFAULT_CONFIGS() noexcept
    {
        return {};
    }

    TransportUARTConsole() noexcept : Transport(Info(1, "UART Console", Type::UART, { DEFAULT_CONFIGS() })) { }

    core::Status init() override
    {
        if (_info.status != Status::Uninitialized)
        {
            return STATUS(ENXIO, "Transport is already initialized or in an invalid state");
        }

        auto ret = usb_serial_jtag_driver_install(&_config);
        if (ret != ESP_OK)
        {
            LOG(ERROR, "Failed to install USB Serial JTAG driver: %s", esp_err_to_name(ret));
            return STATUS(EIO, "Failed to install USB Serial JTAG driver");
        }

        if (!usb_serial_jtag_is_connected())
        {
            LOG(ERROR, "USB Serial JTAG is not connected");
            return STATUS(ENOTCONN, "USB Serial JTAG is not connected");
        }

        if (!_rx_buffer)
        {
            _rx_buffer = core::RingBuffer<std::byte>::create(_config.rx_buffer_size);
            if (!_rx_buffer)
            {
                LOG(ERROR, "Failed to create RX buffer");
                return STATUS(ENOMEM, "Failed to create RX buffer");
            }
        }

        _info.status = Status::Idle;

        return STATUS_OK();
    }

    core::Status deinit() override
    {
        if (_info.status <= Status::Uninitialized)
        {
            return STATUS_OK();
        }

        if (_info.status == Status::Locked)
        {
            return STATUS(EBUSY, "Transport is currently locked");
        }

        auto ret = usb_serial_jtag_driver_uninstall();
        if (ret != ESP_OK)
        {
            LOG(ERROR, "Failed to uninstall USB Serial JTAG driver: %s", esp_err_to_name(ret));
            return STATUS(EIO, "Failed to uninstall USB Serial JTAG driver");
        }

        if (_rx_buffer)
        {
            _rx_buffer.reset();
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    core::Status updateConfig(const core::ConfigMap &configs) override
    {
        return STATUS(ENOTSUP, "Update config is not supported for UART transport");
    }

    inline int available() const override
    {
        if (!_rx_buffer) [[unlikely]]
        {
            return -EFAULT;
        }
        if (!initialized()) [[unlikely]]
        {
            return 0;
        }

        int ret = syncReadBuffer();
        if (ret != 0) [[unlikely]]
        {
            LOG(ERROR, "Failed to sync read buffer: %s", std::strerror(ret));
            return ret;
        }
        return _rx_buffer->size();
    }

    inline int read(void *data, size_t size) override
    {
        if (!_rx_buffer) [[unlikely]]
        {
            return -EFAULT;
        }

        int ret = syncReadBuffer();
        if (ret != 0) [[unlikely]]
        {
            LOG(ERROR, "Failed to sync read buffer: %s", std::strerror(ret));
            return ret;
        }

        return _rx_buffer->read(reinterpret_cast<std::byte *>(data), size);
    }

    inline int write(const void *data, size_t size) override
    {
        if (!initialized()) [[unlikely]]
        {
            return size; // Not initialized, return size as if all data was written
        }
        if (!data || size == 0) [[unlikely]]
        {
            return 0;
        }

        size_t written = 0;
        while (written < size)
        {
            size_t to_write = std::min(size - written, static_cast<size_t>(_config.tx_buffer_size));
            int ret = usb_serial_jtag_write_bytes(static_cast<const unsigned char *>(data) + written, to_write,
                portMAX_DELAY);
            if (ret < 0) [[unlikely]]
            {
                LOG(ERROR, "Failed to write data to USB Serial JTAG: %s", esp_err_to_name(ret));
                return -EIO;
            }
            written += ret;
        }

        return written;
    }

    inline int flush() override
    {
        return fsync(fileno(stdout));
    }

protected:
    inline int syncReadBuffer() const noexcept override
    {
        int r_len = 0;
        std::byte r_buf[32];
        for (;;)
        {
            r_len = usb_serial_jtag_read_bytes(r_buf, sizeof(r_buf), pdMS_TO_TICKS(10));
            if (r_len < 0) [[unlikely]]
            {
                return -EIO;
            }
            else if (r_len == 0)
            {
                break;
            }
            _rx_buffer->write(r_buf, r_len);
        }

        return r_len;
    }

    inline core::RingBuffer<std::byte> &getReadBuffer() noexcept override
    {
        return *_rx_buffer;
    }

private:
    usb_serial_jtag_driver_config_t _config = {
        .tx_buffer_size = 8192,
        .rx_buffer_size = 4096,
    };

    std::unique_ptr<core::RingBuffer<std::byte>> _rx_buffer = nullptr;
};

} // namespace porting

#endif // UART_HPP
