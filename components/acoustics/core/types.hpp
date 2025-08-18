#pragma once
#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace core {

struct __attribute__((packed)) class_t
{
    int id;
    float confidence;
};

}; // namespace core

#endif
