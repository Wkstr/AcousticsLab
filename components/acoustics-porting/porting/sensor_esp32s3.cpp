#include "hal/sensor.hpp"

#include "sensor/i2smic.hpp"
#include "sensor/lis3dhtr.hpp"
#if defined(BOARD_RESPEAKER_LITE) && BOARD_RESPEAKER_LITE
#include "sensor/i2sxmos.hpp"
#endif

namespace bridge {

void __REGISTER_SENSORS__()
{
    [[maybe_unused]] static porting::SensorLIS3DHTR sensor_lis3dhtr;
#if defined(BOARD_RESPEAKER_LITE) && BOARD_RESPEAKER_LITE
    [[maybe_unused]] static porting::SensorI2SXMOS sensor_i2sxmos;
#else
    [[maybe_unused]] static porting::SensorI2SMic sensor_i2smic;
#endif
}

} // namespace bridge
