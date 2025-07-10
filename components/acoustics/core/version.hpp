#pragma once
#ifndef VERSION_HPP
#define VERSION_HPP

#if defined(CORE_VERSION_MAJOR)
#warning "CORE_VERSION_MAJOR is already defined, using existing value"
#else
#define CORE_VERSION_MAJOR 1
#endif

#if defined(CORE_VERSION_MINOR)
#warning "CORE_VERSION_MINOR is already defined, using existing value"
#else
#define CORE_VERSION_MINOR 0
#endif

#if defined(CORE_VERSION_PATCH)
#warning "CORE_VERSION_PATCH is already defined, using existing value"
#else
#define CORE_VERSION_PATCH 0
#endif

#if defined(CORE_VERSION_STRINGIFY)
#warning "CORE_VERSION_STRINGIFY is already defined, rmoving existing definition"
#undef CORE_VERSION_STRINGIFY
#endif

#define CORE_VERSION_STRINGIFY(x) #x

#if defined(CORE_VERSION_TOSTRING)
#warning "CORE_VERSION_TOSTRING is already defined, removing existing definition"
#undef CORE_VERSION_TOSTRING
#endif

#define CORE_VERSION_TOSTRING(x) CORE_VERSION_STRINGIFY(x)

#if defined(CORE_VERSION)
#warning "CORE_VERSION is already defined, removing existing definition"
#undef CORE_VERSION
#endif

#define CORE_VERSION                                                                                                   \
    (CORE_VERSION_TOSTRING(CORE_VERSION_MAJOR) "." CORE_VERSION_TOSTRING(                                              \
        CORE_VERSION_MINOR) "." CORE_VERSION_TOSTRING(CORE_VERSION_PATCH))

#endif
