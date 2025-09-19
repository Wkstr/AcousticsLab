#include "hal/sensor.hpp"

#include "sensor/i2smic.hpp"
#include "sensor/lis3dhtr.hpp"
#if defined(PORTING_BOARD_MODEL_RESPEAKER_LITE) && PORTING_BOARD_MODEL_RESPEAKER_LITE
#include "sensor/i2sxmos.hpp"
#endif

namespace bridge {

void __REGISTER_SENSORS__()
{
    [[maybe_unused]] static porting::SensorLIS3DHTR sensor_lis3dhtr;
#if defined(PORTING_BOARD_MODEL_RESPEAKER_LITE) && PORTING_BOARD_MODEL_RESPEAKER_LITE
    [[maybe_unused]] static porting::SensorI2SXMOS sensor_i2sxmos;
#elif defined(PORTING_BOARD_MODEL_XIAO_S3) && PORTING_BOARD_MODEL_XIAO_S3
    [[maybe_unused]] static porting::SensorI2SMic sensor_i2smic;
#else
    [[maybe_unused]] static porting::SensorI2SMic sensor_i2smic;
#endif
}

} // namespace bridge
