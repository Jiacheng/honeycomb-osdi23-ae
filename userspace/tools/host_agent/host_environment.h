#pragma once

#include "opencl/hsa/platform.h"
#include <absl/status/status.h>
#include <atomic>
#include <string>

namespace ocl::hsa::enclave {

class HostEnvironment {
  public:
    struct Options {
        uintptr_t gtt_vaddr;
        size_t gtt_size;
        uintptr_t vram_vaddr;
        size_t vram_size;
    };
    HostEnvironment();
    absl::Status Open(const std::string &shm_fn, const Options &options,
                      Device *device);
    void Destroy();
    ~HostEnvironment();
    void *GetConfigurationSpaceBase() const { return conf_space_; }

  private:
    absl::Status PrepareConfigurationSpace(Device *dev);
    absl::Status PrepareMapping();
    std::string fn_;

    int fd_;
    Options options_;
    void *conf_space_;
    void *gtt_buf_;
    void *vram_address_space_;
};
} // namespace ocl::hsa::enclave