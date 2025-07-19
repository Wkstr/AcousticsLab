#pragma once
#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <esp_heap_caps.h>

#include "logger.hpp"

namespace core {

inline void printHeapInfo(uint32_t caps = MALLOC_CAP_8BIT) noexcept
{
    multi_heap_info_t heap_info;
    heap_caps_get_info(&heap_info, caps);
    LOG(INFO,
        "Heap info: "
        "used=%zu, "
        "free=%zu, "
        "minimum_free=%zu, "
        "largest_free_block=%zu",
        heap_info.total_allocated_bytes, heap_info.total_free_bytes, heap_info.minimum_free_bytes,
        heap_info.largest_free_block);
}

} // namespace core

#endif
