#! /usr/bin/env python3

# pylint: disable=missing-module-docstring
import argparse
import json
import logging
import os
import sys
import threading
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"

if IS_LINUX or IS_MACOS:
    import termios
    import tty


# pylint: disable=missing-class-docstring,invalid-name
class ResponseType(Enum):
    Direct = 0
    Event = 1
    Stream = 2
    System = 3
    Unknown = 4


# pylint: disable=too-many-instance-attributes
class Client:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        device: str,
        baudrate: int,
        data_buffer_limit: int = 8192,
        response_buffer_limit: int = 32,
    ):
        self._device = device
        self._baudrate = baudrate
        self._data_buffer_limit = data_buffer_limit
        self._response_buffer_limit = response_buffer_limit

        try:
            self._port = self._configure_serial_port()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._port = None
            logging.error("Failed to configure serial port: %s", e)
            raise
        self._write_function = self._configure_write_function()
        self._read_function = self._configure_read_function()
        self._flush = self._configure_flush_function()

        self._data_buffer = ""
        self._response_buffer = {k: [] for k in ResponseType}
        self._response_buffer_locks = {
            k: threading.Lock() for k in self._response_buffer
        }
        self._stop_requested = False

    def __del__(self):
        if self._port is not None:
            os.close(self._port)

    # pylint: disable=too-many-return-statements
    def _parse_response(self):
        RESPONSE_PREFIX = "\r"
        RESPONSE_SUFFIX = "\n"

        if len(self._data_buffer) < len(RESPONSE_PREFIX) + len(RESPONSE_SUFFIX):
            return

        buffer_str = self._data_buffer
        start_index = buffer_str.find(RESPONSE_PREFIX)
        start_index += len(RESPONSE_PREFIX)
        end_index = buffer_str.find(RESPONSE_SUFFIX, start_index)
        if end_index == -1:
            return
        response_str = buffer_str[start_index:end_index]
        end_index += len(RESPONSE_SUFFIX)
        self._data_buffer = buffer_str[end_index:]
        if len(response_str) == 0:
            return
        response_str = response_str.strip()
        if not response_str.startswith("{") or not response_str.endswith("}"):
            logging.info("Plain response: %s", response_str)
            return

        try:
            response_data = json.loads(response_str)
            response_type = ResponseType(
                int(response_data.get("type", ResponseType.Unknown))
            )
            with self._response_buffer_locks[response_type]:
                while (
                    len(self._response_buffer[response_type])
                    >= self._response_buffer_limit
                ):
                    self._response_buffer[response_type].pop(0)
                self._response_buffer[response_type].append(response_data)

        except json.JSONDecodeError:
            logging.warning("Failed to decode JSON response: %s", response_str)
            return
        except ValueError as e:
            logging.warning("Error: %s, response: %s", e, response_str)
            return
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.error("Unexpected error while parsing response: %s", e)
            return
        finally:
            logging.debug("Response: %s", response_str)

    # pylint: disable=missing-function-docstring
    def run(self, buffer_size: int = 8192):
        logging.info("Starting client on %s at %s baud.", self._device, self._baudrate)
        try:
            while not self._stop_requested:
                data = self.read_some(buffer_size)
                if isinstance(data, str):
                    data_len = len(data)
                    if data_len == 0:
                        continue
                    buffer_len = len(self._data_buffer)
                    discard_len = buffer_len + data_len - self._data_buffer_limit
                    if discard_len > 0:
                        self._data_buffer = self._data_buffer[discard_len:]
                    self._data_buffer += data
                    self._parse_response()
        except KeyboardInterrupt:
            logging.info("Client stopped by user.")
        finally:
            self._stop_requested = False

    def stop(self):
        logging.info("Stopping client.")
        self._stop_requested = True

    def clear_responses(self, response_type: ResponseType | None = None):
        if response_type is None:
            for key in self._response_buffer.keys():
                with self._response_buffer_locks[key]:
                    self._response_buffer[key].clear()
        elif response_type in self._response_buffer:
            with self._response_buffer_locks[response_type]:
                self._response_buffer[response_type].clear()
        else:
            raise ValueError(f"Invalid response type: {response_type}")

    def receive_response(
        self, response_type: ResponseType, name: str, tag: str | None = None
    ):
        if response_type not in self._response_buffer:
            raise ValueError(f"Invalid response type: {response_type}")

        result = None
        result_idx = 0
        with self._response_buffer_locks[response_type]:
            for i, response in enumerate(self._response_buffer[response_type]):
                if (tag is None or response.get("tag", "") == tag) and response.get(
                    "name", ""
                ) == name:
                    result = response
                    result_idx = i
                    break
            if result is not None:
                del self._response_buffer[response_type][result_idx]

        return result

    def send_command(self, command: str) -> int:
        if not command.startswith("\n"):
            command = "\n" + command
        if not command.endswith("\n"):
            command += "\n"
        res = self.write_some(command.encode("utf-8"))
        return res

    def write_some(self, data: bytes) -> int:
        return self._write_function(data)

    def read_some(self, buffer_size: int = 8192) -> str | None:
        return self._read_function(buffer_size)

    def flush(self):
        self._flush()

    def _configure_flush_function(self):
        if IS_LINUX or IS_MACOS:
            return lambda: termios.tcflush(self._port, termios.TCOFLUSH)

        raise NotImplementedError(f"Unsupported platform: {sys.platform}")

    def _configure_read_function(self):
        if IS_LINUX or IS_MACOS:
            return lambda buffer_size: os.read(self._port, buffer_size).decode(
                "utf-8", errors="ignore"
            )

        raise NotImplementedError(f"Unsupported platform: {sys.platform}")

    def _configure_write_function(self):
        if IS_LINUX or IS_MACOS:
            return lambda data: os.write(self._port, data)

        raise NotImplementedError(f"Unsupported platform: {sys.platform}")

    def _configure_serial_port(self):
        if IS_LINUX or IS_MACOS:
            BAUD_MAP = {
                9600: termios.B9600,
                19200: termios.B19200,
                38400: termios.B38400,
                57600: termios.B57600,
                115200: termios.B115200,
            }
            if IS_LINUX:
                BAUD_MAP.update({921600: termios.B921600})  # pylint: disable=no-member
            if self._baudrate not in BAUD_MAP:
                raise ValueError(f"Unsupported baudrate: {self._baudrate}")

            fd = os.open(self._device, os.O_RDWR | os.O_NOCTTY | os.O_NDELAY)

            tty.setraw(fd)

            attrs = termios.tcgetattr(fd)
            attrs[tty.IFLAG] &= ~(
                termios.IGNBRK
                | termios.BRKINT
                | termios.PARMRK
                | termios.ISTRIP
                | termios.INLCR
                | termios.IGNCR
                | termios.ICRNL
                | termios.IXON
            )
            attrs[tty.OFLAG] &= ~termios.OPOST
            attrs[tty.LFLAG] &= ~(
                termios.ECHO
                | termios.ECHONL
                | termios.ECHOCTL
                | termios.ICANON
                | termios.ISIG
                | termios.IEXTEN
            )
            attrs[tty.CFLAG] &= ~(termios.CSIZE | termios.PARENB)
            attrs[tty.CFLAG] |= termios.CS8
            attrs[tty.CFLAG] |= termios.CREAD | termios.CLOCAL
            attrs[tty.ISPEED] |= BAUD_MAP[self._baudrate]
            attrs[tty.OSPEED] |= BAUD_MAP[self._baudrate]
            attrs[tty.CC][termios.VMIN] = 0
            attrs[tty.CC][termios.VTIME] = 0

            termios.tcsetattr(fd, termios.TCSANOW, attrs)

            return fd

        raise NotImplementedError(f"Unsupported platform: {sys.platform}")


def main():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser(
        description="Send and receive data over a serial port."
    )
    parser.add_argument(
        "--device",
        default="/dev/ttyACM0",
        help="Serial device device name (e.g. /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=921600,
        help="Baud rate for the serial communication (e.g. 921600)",
    )
    parser.add_argument(
        "--data-buffer-limit",
        type=int,
        default=8192,
        help="Maximum data buffer limit",
    )
    parser.add_argument(
        "--response-buffer-limit",
        type=int,
        default=32,
        help="Maximum response buffer limit",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    client = Client(
        args.device,
        args.baudrate,
        args.data_buffer_limit,
        args.response_buffer_limit,
    )
    client_thread = threading.Thread(target=client.run, daemon=True)
    client_thread.start()

    try:
        while True:
            user_input = input()
            if user_input.lower() == "exit":
                break
            if user_input is not None:
                client.send_command(user_input)
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error("An error occurred: %s", e)
    finally:
        client.stop()
        client_thread.join()


if __name__ == "__main__":
    main()
