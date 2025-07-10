#include "device.hpp"

namespace hal {

core::Status DeviceRegistry::registerDevice(Device *device) noexcept
{
    if (!device)
    {
        return STATUS(EINVAL, "Device cannot be null");
    }
    if (_device)
    {
        return STATUS(EEXIST, "Device already registered");
    }
    _device = device;
    return STATUS_OK();
}

Device *DeviceRegistry::_device = nullptr;

} // namespace hal
