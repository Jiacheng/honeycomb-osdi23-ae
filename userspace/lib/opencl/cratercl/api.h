#pragma once

#include "icd_dispatch.h"
#include <CL/cl.h>
#include <cstring>
#include <tuple>
#include <utility>

namespace crater::opencl {

namespace detail {

template <typename T> struct ParamInfo {
    static inline std::pair<const void *, size_t> get(const T &param) {
        return std::pair<const void *, size_t>(&param, sizeof(T));
    }
};

template <> struct ParamInfo<const char *> {
    static inline std::pair<const void *, size_t> get(const char *param) {
        return std::pair<const void *, size_t>(param, strlen(param) + 1);
    }
};

template <int N> struct ParamInfo<char[N]> {
    static inline std::pair<const void *, size_t> get(const char *param) {
        return std::pair<const void *, size_t>(param, strlen(param) + 1);
    }
};

} // namespace detail

extern const _cl_icd_dispatch kCLDispatchTable;
template <class T> struct CLObject : public T {
  protected:
    CLObject() {
        static_cast<T *>(this)->dispatch =
            const_cast<cl_icd_dispatch *>(&kCLDispatchTable);
    }
};

template <typename T>
static inline cl_int clGetInfo(T &field, size_t param_value_size,
                               void *param_value,
                               size_t *param_value_size_ret) {
    const void *valuePtr;
    size_t valueSize;

    std::tie(valuePtr, valueSize) =
        detail::ParamInfo<typename std::remove_const<T>::type>::get(field);

    if (param_value_size_ret) {
        *param_value_size_ret = valueSize;
    }

    cl_int ret = CL_SUCCESS;
    if (param_value != NULL && param_value_size < valueSize) {
        if (!std::is_pointer<T>() ||
            !std::is_same<typename std::remove_const<
                              typename std::remove_pointer<T>::type>::type,
                          char>()) {
            return CL_INVALID_VALUE;
        }
        // For char* and char[] params, we will at least fill up to
        // param_value_size, then return an error.
        valueSize = param_value_size;
        static_cast<char *>(param_value)[--valueSize] = '\0';
        ret = CL_INVALID_VALUE;
    }

    if (param_value != NULL) {
        ::memcpy(param_value, valuePtr, valueSize);
        if (param_value_size > valueSize) {
            ::memset(static_cast<char *>(param_value) + valueSize, '\0',
                     param_value_size - valueSize);
        }
    }

    return ret;
}

template <class T> struct Nullable {
    explicit Nullable(T *ptr) : ptr_(ptr) {}
    void Assign(const T &value) {
        if (ptr_) {
            *ptr_ = value;
        }
    }

  private:
    T *const ptr_;
};

extern "C" {
__attribute__((visibility("default"))) void *
clGetExtensionFunctionAddress(const char *func_name);
}
} // namespace crater::opencl
