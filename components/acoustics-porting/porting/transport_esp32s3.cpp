#include "hal/transport.hpp"

#include "transport/uart_console.hpp"

namespace bridge {

void __REGISTER_TRANSPORTS__()
{
    [[maybe_unused]] static porting::TransportUARTConsole transport_uart_console;
}

} // namespace bridge
