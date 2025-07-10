#include "transport.hpp"

namespace hal {

core::Status TransportRegistry::registerTransport(Transport *transport) noexcept
{
    if (!transport)
    {
        return STATUS(EINVAL, "Transport is null");
    }
    if (_transports.contains(transport->info().id))
    {
        return STATUS(EEXIST, "Transport with ID already exists");
    }

    _transports[transport->info().id] = transport;

    return STATUS_OK();
}

TransportRegistry::TransportMap TransportRegistry::_transports = {};

} // namespace hal
