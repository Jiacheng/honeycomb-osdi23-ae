#pragma once

#include <cerrno>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>

namespace ocl::hsa {

static inline int kmtIoctl(int fd, unsigned long request, void *arg) {
    int ret;

    do {
        ret = ioctl(fd, request, arg);
    } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

    if (ret == -1 && errno == EBADF) {
        spdlog::warn("Invalid FD");
    }

    return ret;
}

static inline unsigned lower_32_bits(unsigned long v) { return (unsigned)v; }
static inline unsigned upper_32_bits(unsigned long v) {
    return (unsigned)(v >> 32);
}

} // namespace ocl::hsa
