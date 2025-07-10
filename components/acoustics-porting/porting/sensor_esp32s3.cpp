#include "hal/sensor.hpp"

#include "sensor/lis3dhtr.hpp"

namespace bridge {

void __REGISTER_SENSORS__()
{
    [[maybe_unused]] static porting::SensorLIS3DHTR sensor_lis3dhtr;
}

} // namespace bridge
