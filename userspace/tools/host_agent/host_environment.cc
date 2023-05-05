#include "host_environment.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/enclave/idl.h"
#include "opencl/hsa/platform.h"
#include <absl/status/status.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace ocl::hsa::enclave {

enum {
    kConfigurationSpaceSize =
        idl::ConfigurationSpaceLayout::kConfigurationSpaceSize,
};

HostEnvironment::HostEnvironment()
    : fd_(-1), gtt_buf_(nullptr), vram_address_space_(nullptr) {}

absl::Status HostEnvironment::Open(const std::string &shm_fn,
                                   const Options &options, Device *dev) {
    fd_ = open(shm_fn.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd_ < 0) {
        return absl::InvalidArgumentError("Cannot open file");
    }
    fn_ = shm_fn;
    options_ = options;
    if (options.gtt_vaddr % Device::kHugeGPUPageSize != 0 ||
        options.gtt_size % Device::kHugeGPUPageSize != 0 ||
        options.vram_vaddr % Device::kHugeGPUPageSize != 0 ||
        options.vram_size % Device::kHugeGPUPageSize != 0) {
        return absl::InvalidArgumentError("The address space and the size have "
                                          "to be aligned on 2MB boundaries");
    }
    // Only the configuration space and the GTT have to be realized shared
    // memory. We only need to reserve the address space for VRAM.
    auto total_size = kConfigurationSpaceSize + options.gtt_size;
    if (ftruncate(fd_, total_size)) {
        return absl::InvalidArgumentError(
            "Cannot set the size of the shared memory buffer");
    }

    auto stat = PrepareMapping();
    if (!stat.ok()) {
        return stat;
    }

    stat = PrepareConfigurationSpace(dev);
    if (!stat.ok()) {
        return stat;
    }
    return absl::OkStatus();
}

absl::Status HostEnvironment::PrepareMapping() {
    gtt_buf_ = mmap(reinterpret_cast<void *>(options_.gtt_vaddr),
                    options_.gtt_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, fd_, kConfigurationSpaceSize);
    if (gtt_buf_ == MAP_FAILED) {
        return absl::InvalidArgumentError(
            "Cannot reserve the GTT memory space");
    }

    vram_address_space_ =
        mmap(reinterpret_cast<void *>(options_.vram_vaddr), options_.vram_size,
             PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE, -1, 0);
    if (vram_address_space_ == MAP_FAILED) {
        return absl::InvalidArgumentError(
            "Cannot reserve the VRAM memory space");
    }
    return absl::OkStatus();
}

absl::Status HostEnvironment::PrepareConfigurationSpace(Device *dev) {
    conf_space_ = mmap(nullptr, kConfigurationSpaceSize, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd_, 0);
    if (conf_space_ == MAP_FAILED) {
        return absl::InvalidArgumentError(
            "Cannot initialize configuration space");
    }
    idl::HostConfiguration conf = {
        .gtt_vaddr = options_.gtt_vaddr,
        .gtt_size = options_.gtt_size,
        .vram_vaddr = options_.vram_vaddr,
        .vram_size = options_.vram_size,
        .node_id = dev->GetNodeID(),
        .gpu_id = dev->GetGPUID(),
    };
    *reinterpret_cast<idl::HostConfiguration *>(conf_space_) = conf;
    return absl::OkStatus();
}

void HostEnvironment::Destroy() {
    if (fd_ < 0) {
        return;
    }
    munmap(conf_space_, kConfigurationSpaceSize);
    munmap(gtt_buf_, options_.gtt_size);
    munmap(vram_address_space_, options_.vram_size);
    close(fd_);
    unlink(fn_.c_str());
    fd_ = -1;
}

HostEnvironment::~HostEnvironment() { Destroy(); }
} // namespace ocl::hsa::enclave
