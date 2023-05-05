#pragma once

#include "runtime-gpu/core/abi.h"
#include <initializer_list>

namespace gpumpc {

template <class T, unsigned N> struct Array {
    using value_type = T;
    GPUMPC_HOST_AND_DEVICE inline value_type &operator[](unsigned index) {
        return data_[index];
    }
    GPUMPC_HOST_AND_DEVICE inline const value_type &
    operator[](unsigned index) const {
        return data_[index];
    }
    constexpr unsigned size() const { return N; }
    GPUMPC_HOST_AND_DEVICE inline T *data() { return data_; }
    GPUMPC_HOST_AND_DEVICE inline const T *data() const { return data_; }
    T data_[N];
};
} // namespace gpumpc