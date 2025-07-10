#pragma once
#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <cstdio>
#include <chrono>

namespace core {

#if defined(VERBOSE) || defined(DEBUG) || defined(INFO) || defined(WARNING) || defined(ERROR) || defined(NONE)
#warning "Log level macros may conflict with existing definitions. Please ensure they are unique."
#endif

#define VERBOSE 5
#define DEBUG   4
#define INFO    3
#define WARNING 2
#define ERROR   1
#define NONE    0

#ifndef LOG_LEVEL
#define LOG_LEVEL WARNING
#elif LOG_LEVEL < NONE || LOG_LEVEL > VERBOSE
#error "LOG_LEVEL must be between NONE (0) and VERBOSE (5)"
#endif

#define LOG(level, messages...)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr (level <= LOG_LEVEL)                                                                              \
        {                                                                                                              \
            printf("[%c] ", #level[0]);                                                                                \
            if constexpr (LOG_LEVEL >= DEBUG)                                                                          \
            {                                                                                                          \
                printf("[%lld] ", std::chrono::duration_cast<std::chrono::milliseconds>(                               \
                                      std::chrono::system_clock::now().time_since_epoch())                             \
                                      .count());                                                                       \
            }                                                                                                          \
            if constexpr (LOG_LEVEL >= VERBOSE)                                                                        \
            {                                                                                                          \
                printf("%s:%d ", __FILE__, __LINE__);                                                                  \
            }                                                                                                          \
            if constexpr (LOG_LEVEL >= DEBUG)                                                                          \
            {                                                                                                          \
                printf("%s: ", __FUNCTION__);                                                                          \
            }                                                                                                          \
            printf(messages);                                                                                          \
            printf("\n");                                                                                              \
        }                                                                                                              \
    }                                                                                                                  \
    while (0)

} // namespace core

#endif
