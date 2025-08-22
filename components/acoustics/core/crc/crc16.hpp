#pragma once
#ifndef CRC16_HPP
#define CRC16_HPP

#include "core/crc.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace core {

namespace crc::detail {

    class CRC16XMODEM final: public CRC<uint16_t, CRC16XMODEM>
    {
    public:
        CRC16XMODEM() noexcept : CRC(static_cast<uint16_t>(0x0000)) { }

        void update(const uint8_t *data, size_t size) noexcept
        {
            if (!data || size == 0) [[unlikely]]
            {
                return;
            }

            for (size_t i = 0; i < size; ++i)
            {
                _crc = (_crc << 8) ^ _table[(_crc >> 8) ^ data[i]];
            }
        }

    private:
        static constexpr inline std::array<uint16_t, 256> _table = [] {
            std::array<uint16_t, 256> table {};
            for (size_t i = 0; i < 256; ++i)
            {
                uint16_t crc = static_cast<uint16_t>(i << 8);
                for (size_t j = 0; j < 8; ++j)
                {
                    if (crc & 0x8000)
                    {
                        crc = (crc << 1) ^ 0x1021;
                    }
                    else
                    {
                        crc <<= 1;
                    }
                }
                table[i] = crc;
            }
            return table;
        }();
    };

} // namespace crc::detail

using CRC16XMODEM = crc::detail::CRC16XMODEM;

} // namespace core

#endif
