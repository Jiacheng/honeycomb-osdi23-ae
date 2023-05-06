#include "guest_platform.h"
#include "guest_rpc_client.h"
#include "idl.h"
#include "opencl/hsa/runtime_options.h"
#include "transmit_buffer.h"

#include <absl/status/status.h>
#include <atomic>
#include <cstddef>
#include <memory>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/un.h>
#include <unistd.h>

namespace ocl::hsa::enclave {

using idl::ConfigurationSpaceLayout;

std::unique_ptr<MemoryManager>
NewEnclaveGuestMemoryManager(Device *dev, absl::Span<char> gtt_space,
                             absl::Span<char> vram_space);
std::unique_ptr<Event> NewEnclaveGuestEvent(EnclaveGuestDevice *dev, int type,
                                            uint64_t event_page_handle);

EnclaveGuestPlatform::EnclaveGuestPlatform()
    : shm_fd_(-1), sock_fd_(-1), agent_physical_memory_fd_(-1) {}

Platform &EnclaveGuestPlatform::Instance() {
    static EnclaveGuestPlatform inst;
    return inst;
}

void EnclaveGuestPlatform::SetOptions(const Options &options) {
    options_ = options;
}

absl::Status EnclaveGuestPlatform::Initialize() { return PrepareDevices(); }

absl::Status EnclaveGuestPlatform::PrepareDevices() {
    int shm_fd = open(options_.shm_fn.c_str(), O_RDWR);
    if (!shm_fd) {
        return absl::InvalidArgumentError("Cannot open the shared memory");
    }
    conf_space_ =
        mmap(nullptr, ConfigurationSpaceLayout::kConfigurationSpaceSize,
             PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (conf_space_ == MAP_FAILED) {
        close(shm_fd);
        return absl::InvalidArgumentError("Cannot map in the shared memory");
    }
    config_ = *reinterpret_cast<const idl::HostConfiguration *>(conf_space_);
    std::unique_ptr<Device> dev;
    auto base = reinterpret_cast<char *>(conf_space_);
    auto stat = CreateDevice(shm_fd, base, &config_, &dev);
    if (!stat.ok()) {
        close(shm_fd);
        return stat;
    }
    stat = dev->Initialize();
    if (!stat.ok()) {
        close(shm_fd);
        return stat;
    }

    shm_fd_ = shm_fd;
    devices_.emplace_back(dev.release());
    auto opt = GetRuntimeOptions();
    if (opt->MapRemotePhysicalPage()) {
        agent_physical_memory_fd_ =
            open(opt->GetAgentPhysicalMemoryPath().c_str(), O_RDWR);
        if (agent_physical_memory_fd_ < 0) {
            return absl::InvalidArgumentError(
                "Cannot open the agent physical memory");
        }
    }

    return absl::OkStatus();
}

absl::Status EnclaveGuestPlatform::CreateDevice(int shm_fd, char *base,
                                                idl::HostConfiguration *config,
                                                std::unique_ptr<Device> *ret) {
    if (!GetRuntimeOptions()->MapRemotePhysicalPage()) {
        auto gtt_vaddr =
            mmap(reinterpret_cast<void *>(config->gtt_vaddr), config->gtt_size,
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, shm_fd,
                 ConfigurationSpaceLayout::kConfigurationSpaceSize);
        if (gtt_vaddr == MAP_FAILED) {
            return absl::InvalidArgumentError("Cannot map in the GTT memory");
        }
    }

    auto watermark_rx =
        reinterpret_cast<idl::ConfigurationSpaceLayout::Watermark *>(
            base + ConfigurationSpaceLayout::kRXBufferWatermarkOffset);
    auto watermark_tx =
        reinterpret_cast<idl::ConfigurationSpaceLayout::Watermark *>(
            base + ConfigurationSpaceLayout::kTXBufferWatermarkOffset);
    auto rx_buf = base + ConfigurationSpaceLayout::kRXBufferOffset;
    auto tx_buf = base + ConfigurationSpaceLayout::kTXBufferOffset;
    TransmitBuffer host_tx(
        absl::MakeSpan(tx_buf, ConfigurationSpaceLayout::kTransmitBufferSize),
        reinterpret_cast<std::atomic_size_t *>(&watermark_tx->rptr),
        reinterpret_cast<std::atomic_size_t *>(&watermark_tx->wptr));
    TransmitBuffer host_rx(
        absl::MakeSpan(rx_buf, ConfigurationSpaceLayout::kTransmitBufferSize),
        reinterpret_cast<std::atomic_size_t *>(&watermark_rx->rptr),
        reinterpret_cast<std::atomic_size_t *>(&watermark_rx->wptr));

    std::unique_ptr<EnclaveGuestDevice> dev(
        new EnclaveGuestDevice(config->node_id, config->gpu_id,
                               std::move(host_rx), std::move(host_tx)));
    dev->mm_ = NewEnclaveGuestMemoryManager(
        dev.get(),
        absl::MakeSpan(reinterpret_cast<char *>(config->gtt_vaddr),
                       config->gtt_size),
        absl::MakeSpan(reinterpret_cast<char *>(config->vram_vaddr),
                       config->vram_size));

    *ret = std::move(dev);
    return absl::OkStatus();
}

absl::Status EnclaveGuestPlatform::NotifyHostAgent() {
    // TODO
    // auto ret = read(shm_fd_, NULL, 0);
    // (void)ret;
    return absl::OkStatus();
}

absl::Status EnclaveGuestPlatform::Close() {
    if (sock_fd_ < 0 || shm_fd_ < 0) {
        return absl::OkStatus();
    }
    close(shm_fd_);
    close(sock_fd_);
    sock_fd_ = -1;
    return absl::OkStatus();
}

std::unique_ptr<Event>
EnclaveGuestPlatform::NewEvent(int type, uint64_t event_page_handle) {
    auto dev = static_cast<EnclaveGuestDevice *>(devices_.front());
    return NewEnclaveGuestEvent(dev, type, event_page_handle);
}

} // namespace ocl::hsa::enclave
