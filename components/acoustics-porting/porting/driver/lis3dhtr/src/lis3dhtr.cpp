#include "lis3dhtr.h"

#include <freertos/FreeRTOS.h>
#include <freertos/timers.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace driver {

#define LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_2G  (0.001f)
#define LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_4G  (0.002f)
#define LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_8G  (0.004f)
#define LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_16G (0.012f)

#define LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_2G  (0.004f)
#define LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_4G  (0.008f)
#define LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_8G  (0.016f)
#define LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_16G (0.048f)

#define LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_2G  (0.016f)
#define LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_4G  (0.032f)
#define LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_8G  (0.064f)
#define LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_16G (0.192f)

LIS3DHTR::LIS3DHTR()
{
    _i2c_mst_config = {};
    _bus_handle = nullptr;
    _dev_config = {};
    _dev_handle = nullptr;

    _power_mode = LIS3DHTR_REG_ACCEL_CTRL_REG1_LPEN_NORMAL;
    _full_scale_range = LIS3DHTR_REG_ACCEL_CTRL_REG4_FS_4G;
    _high_solution = LIS3DHTR_REG_ACCEL_CTRL_REG4_HS_ENABLE;

    _accel_shift = 0;
    _accel_scale = 0.0f;
}

LIS3DHTR::~LIS3DHTR()
{
    deinit();
}

int LIS3DHTR::init(int sda_pin, int scl_pin, i2c_port_num_t i2c_port, size_t clk_speed_hz, int timeout_ms)
{
    if (_bus_handle && _dev_handle)
    {
        return EALREADY;
    }

    _i2c_mst_config = {
        .i2c_port = i2c_port,
        .sda_io_num = static_cast<gpio_num_t>(sda_pin),
        .scl_io_num = static_cast<gpio_num_t>(scl_pin),
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .intr_priority = 0,
        .trans_queue_depth = 0,
        .flags = {
            .enable_internal_pullup = true,
            .allow_pd = false,
        },
    };
    int ret = i2c_new_master_bus(&_i2c_mst_config, &_bus_handle);
    if (ret != ESP_OK)
    {
        deinit();
        return EIO;
    }

    _dev_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = LIS3DHTR_ADDRESS_UPDATED,
        .scl_speed_hz = clk_speed_hz,
        .scl_wait_us = 0,
        .flags = {
            .disable_ack_check = false,
        },
    };
    ret = i2c_master_bus_add_device(_bus_handle, &_dev_config, &_dev_handle);
    if (ret != ESP_OK)
    {
        deinit();
        return EIO;
    }

    ret = i2c_master_probe(_bus_handle, _dev_config.device_address, timeout_ms);
    if (ret != ESP_OK)
    {
        deinit();
        return ENODEV;
    }

    uint8_t lis3dhtr_cfg;

    lis3dhtr_cfg = LIS3DHTR_REG_TEMP_ADC_PD_ENABLED | LIS3DHTR_REG_TEMP_TEMP_EN_DISABLED;
    ret = _writeRegister(LIS3DHTR_REG_TEMP_CFG, &lis3dhtr_cfg, sizeof(lis3dhtr_cfg), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    lis3dhtr_cfg = _power_mode | LIS3DHTR_REG_ACCEL_CTRL_REG1_AZEN_ENABLE | LIS3DHTR_REG_ACCEL_CTRL_REG1_AYEN_ENABLE
                   | LIS3DHTR_REG_ACCEL_CTRL_REG1_AXEN_ENABLE;
    ret = _writeRegister(LIS3DHTR_REG_ACCEL_CTRL_REG1, &lis3dhtr_cfg, sizeof(lis3dhtr_cfg), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    lis3dhtr_cfg = LIS3DHTR_REG_ACCEL_CTRL_REG4_BDU_NOTUPDATED | LIS3DHTR_REG_ACCEL_CTRL_REG4_BLE_LSB | _high_solution
                   | LIS3DHTR_REG_ACCEL_CTRL_REG4_ST_NORMAL | LIS3DHTR_REG_ACCEL_CTRL_REG4_SIM_4WIRE;
    ret = _writeRegister(LIS3DHTR_REG_ACCEL_CTRL_REG4, &lis3dhtr_cfg, sizeof(lis3dhtr_cfg), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    ret = setFullScaleRange(static_cast<lis3dhtr_scale_type_t>(_full_scale_range), timeout_ms);
    if (ret != 0)
    {
        goto err;
    }

    ret = setOutputDataRate(LIS3DHTR_DATARATE_200HZ, timeout_ms);
    if (ret != 0)
    {
        goto err;
    }

    return 0;

err:
    deinit();
    return ret;
}

void LIS3DHTR::deinit()
{
    if (_dev_handle)
    {
        i2c_master_bus_rm_device(_dev_handle);
        _dev_handle = nullptr;
    }
    if (_bus_handle)
    {
        i2c_del_master_bus(_bus_handle);
        _bus_handle = nullptr;
    }
}

int LIS3DHTR::setPowerMode(lis3dhtr_power_type_t mode, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    uint8_t data = 0;
    int ret = _readRegister(LIS3DHTR_REG_ACCEL_CTRL_REG1, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }

    data &= ~LIS3DHTR_REG_ACCEL_CTRL_REG1_LPEN_MASK;
    data |= mode;

    ret = _writeRegister(LIS3DHTR_REG_ACCEL_CTRL_REG1, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    _power_mode = mode;
    _updateAccelShiftAndScale();

    return 0;

err:
    return EIO;
}

int LIS3DHTR::setFullScaleRange(lis3dhtr_scale_type_t range, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    uint8_t data = 0;
    int ret = _readRegister(LIS3DHTR_REG_ACCEL_CTRL_REG4, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }

    data &= ~LIS3DHTR_REG_ACCEL_CTRL_REG4_FS_MASK;
    data |= range;

    ret = _writeRegister(LIS3DHTR_REG_ACCEL_CTRL_REG4, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    _full_scale_range = range;
    _updateAccelShiftAndScale();

    return 0;

err:
    return EIO;
}

int LIS3DHTR::setOutputDataRate(lis3dhtr_odr_type_t odr, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    uint8_t data = 0;
    int ret = _readRegister(LIS3DHTR_REG_ACCEL_CTRL_REG1, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }

    data &= ~LIS3DHTR_REG_ACCEL_CTRL_REG1_AODR_MASK;
    data |= odr;

    ret = _writeRegister(LIS3DHTR_REG_ACCEL_CTRL_REG1, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    return 0;

err:
    return EIO;
}

int LIS3DHTR::setHighSolution(bool enable, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    uint8_t data = 0;
    int ret = _readRegister(LIS3DHTR_REG_ACCEL_CTRL_REG4, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }

    data = enable ? data | LIS3DHTR_REG_ACCEL_CTRL_REG4_HS_ENABLE : data & ~LIS3DHTR_REG_ACCEL_CTRL_REG4_HS_ENABLE;

    ret = _writeRegister(LIS3DHTR_REG_ACCEL_CTRL_REG4, &data, sizeof(data), timeout_ms);
    if (ret != ESP_OK)
    {
        goto err;
    }
    vTaskDelay(pdMS_TO_TICKS(LIS3DHTR_CONVERSIONDELAY));

    _high_solution = data & LIS3DHTR_REG_ACCEL_CTRL_REG4_HS_MASK;
    _updateAccelShiftAndScale();

    return 0;

err:
    return EIO;
}

int LIS3DHTR::getAcceleration(float &x, float &y, float &z, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    uint8_t data[6] = { 0 };

    int ret = _readRegister(LIS3DHTR_REG_ACCEL_OUT_X_L, data, 6, timeout_ms);
    if (ret != ESP_OK)
    {
        return EIO;
    }

    x = static_cast<float>(static_cast<int16_t>((data[1] << 8) | data[0]) >> _accel_shift) * _accel_scale;
    y = static_cast<float>(static_cast<int16_t>((data[3] << 8) | data[2]) >> _accel_shift) * _accel_scale;
    z = static_cast<float>(static_cast<int16_t>((data[5] << 8) | data[4]) >> _accel_shift) * _accel_scale;

    return 0;
}

int LIS3DHTR::getTemperature(int16_t &temperature, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    uint8_t data[2] = { 0 };

    int ret = _readRegister(0x0C, data, 2, timeout_ms);
    if (ret != ESP_OK)
    {
        return EIO;
    }

    temperature = roundf(static_cast<int16_t>((data[1] << 8) | data[0]) / 256.0f) + 25.0f;

    return 0;
}

int LIS3DHTR::getDeviceID(uint8_t &device_id, int timeout_ms)
{
    if (!_dev_handle)
    {
        return ENODEV;
    }

    int ret = _readRegister(LIS3DHTR_REG_ACCEL_WHO_AM_I, &device_id, sizeof(device_id), timeout_ms);
    if (ret != ESP_OK)
    {
        return EIO;
    }

    return 0;
}

inline esp_err_t LIS3DHTR::_writeRegister(uint8_t reg, const uint8_t *data, size_t len, int timeout_ms)
{
    uint8_t buffer[32] = { 0 };
    buffer[0] = reg;
    if (len > sizeof(buffer) - 1) [[unlikely]]
    {
        return EOVERFLOW;
    }
    memcpy(&buffer[1], data, len);

    return i2c_master_transmit(_dev_handle, buffer, len + 1, timeout_ms);
}

inline esp_err_t LIS3DHTR::_readRegister(uint8_t reg, uint8_t *data, size_t len, int timeout_ms)
{
    if (len > 32) [[unlikely]]
    {
        return EOVERFLOW;
    }

    reg |= 0x80;

    return i2c_master_transmit_receive(_dev_handle, &reg, 1, data, len, timeout_ms);
}

void LIS3DHTR::_updateAccelShiftAndScale()
{
    if (_high_solution == LIS3DHTR_REG_ACCEL_CTRL_REG4_HS_DISABLE)
    {
        switch (_power_mode)
        {
            case LIS3DHTR_REG_ACCEL_CTRL_REG1_LPEN_LOW:
                switch (_full_scale_range)
                {
                    case LIS3DHTR_RANGE_2G:
                        _accel_shift = 8;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_2G;
                        break;
                    case LIS3DHTR_RANGE_4G:
                        _accel_shift = 8;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_4G;
                        break;
                    case LIS3DHTR_RANGE_8G:
                        _accel_shift = 8;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_8G;
                        break;
                    case LIS3DHTR_RANGE_16G:
                        _accel_shift = 8;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_LOW_PWR_RANGE_16G;
                        break;
                    default:
                        break;
                }
                break;

            case LIS3DHTR_REG_ACCEL_CTRL_REG1_LPEN_NORMAL:
                switch (_full_scale_range)
                {
                    case LIS3DHTR_RANGE_2G:
                        _accel_shift = 6;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_2G;
                        break;
                    case LIS3DHTR_RANGE_4G:
                        _accel_shift = 6;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_4G;
                        break;
                    case LIS3DHTR_RANGE_8G:
                        _accel_shift = 6;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_8G;
                        break;
                    case LIS3DHTR_RANGE_16G:
                        _accel_shift = 6;
                        _accel_scale = LIS3DHTR_ACCEL_SCALE_NORMAL_RANGE_16G;
                        break;
                    default:
                        break;
                }
                break;
        };
    }
    else
    {
        switch (_full_scale_range)
        {
            case LIS3DHTR_RANGE_2G:
                _accel_shift = 4;
                _accel_scale = LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_2G;
                break;
            case LIS3DHTR_RANGE_4G:
                _accel_shift = 4;
                _accel_scale = LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_4G;
                break;
            case LIS3DHTR_RANGE_8G:
                _accel_shift = 4;
                _accel_scale = LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_8G;
                break;
            case LIS3DHTR_RANGE_16G:
                _accel_shift = 4;
                _accel_scale = LIS3DHTR_ACCEL_SCALE_HIGH_RES_RANGE_16G;
                break;
            default:
                break;
        }
    }
}

} // namespace driver
