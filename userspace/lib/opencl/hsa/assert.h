#pragma once

#include <stdexcept>

#define HSA_ASSERT(x)                                                          \
    do {                                                                       \
        if (!(x))                                                                \
            throw std::invalid_argument("Assertion failed");                   \
    } while (0)
