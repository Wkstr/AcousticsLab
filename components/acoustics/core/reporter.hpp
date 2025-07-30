#pragma once
#ifndef REPORTER_HPP
#define REPORTER_HPP

#include <chrono>
#include <string>
#include <unordered_map>

namespace core {

struct Reporter final
{
    Reporter() = default;
    ~Reporter() = default;

    std::unordered_map<std::string, std::chrono::microseconds> time_micro;
};

} // namespace core

#endif
