#pragma once
#ifndef DATA_FRAME_HPP
#define DATA_FRAME_HPP

#include "logger.hpp"

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <utility>

namespace core {

template<typename T>
struct DataFrame final
{
    using Timestamp = std::chrono::time_point<std::chrono::steady_clock>;
    using Index = size_t;
    using DataType = T;

    DataFrame() : timestamp(), index(0), data() { }

    ~DataFrame() = default;

    Timestamp timestamp;
    Index index;
    DataType data;
};

} // namespace core

#endif // DATA_FRAME_HPP
