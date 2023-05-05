#pragma once

#include <cstddef>
#include <cstdint>

namespace gpumpc {

template <class T> static inline T AlignUp(T value, size_t boundary) {
    return (T)(((uintptr_t)value + boundary - 1) / boundary * boundary);
}

template <class T> static inline T AlignDown(T value, size_t boundary) {
    return (T)((uintptr_t)value / boundary * boundary);
}

} // namespace gpumpc
