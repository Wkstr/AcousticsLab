#include "sensor.hpp"

namespace hal {

core::Status SensorRegistry::registerSensor(Sensor *sensor) noexcept
{
    if (!sensor)
    {
        return STATUS(EINVAL, "Sensor cannot be null");
    }
    if (_sensors.contains(sensor->info().id))
    {
        return STATUS(EEXIST, "Sensor with this ID already exists");
    }

    _sensors[sensor->info().id] = sensor;

    return STATUS_OK();
}

SensorRegistry::SensorMap SensorRegistry::_sensors = {};

} // namespace hal
