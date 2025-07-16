#include "hal/sensor.hpp"

#include "sensor/lis3dhtr.hpp"
#include "sensor/microphone.hpp"

namespace bridge {

void __REGISTER_SENSORS__()
{
    [[maybe_unused]] static porting::SensorLIS3DHTR sensor_lis3dhtr;
    [[maybe_unused]] static porting::SensorMicrophone sensor_microphone;
}

} // namespace bridge
