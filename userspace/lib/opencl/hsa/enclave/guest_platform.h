#pragma once

#include "idl.h"
#include "transmit_buffer.h"

#include "opencl/hsa/assert.h"
#include "opencl/hsa/kfd/kfd_platform.h"
#include <memory>

namespace ocl::hsa::enclave {

class GuestRPCClient;
class EnclaveGuestDevice;

class EnclaveGuestPlatform : public Platform {
  public:
    friend class Platform;
    friend class EnclaveGuestDevice;
    struct Options {
        std::string shm_fn;
    };
    void SetOptions(const Options &options);

    virtual int GetKFDFD() const override {
        HSA_ASSERT(0 && "Unreachable");
        return 0;
    }

    virtual const std::vector<Device *> &GetDevices() const override {
        return devices_;
    }

    virtual gpu_addr_t GetEventPageBase() override {
        HSA_ASSERT(0 && "Unreachable");
        return 0;
    }
    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    virtual std::unique_ptr<Event>
    NewEvent(int type, uint64_t event_page_handle) override;

  protected:
    EnclaveGuestPlatform();
    static Platform &Instance();
    absl::Status CreateDevice(int shm_fd, char *config_space_base,
                              idl::HostConfiguration *config,
                              std::unique_ptr<Device> *dev);
    absl::Status PrepareDevices();
    absl::Status NotifyHostAgent();

    Options options_;
    idl::HostConfiguration config_;
    int shm_fd_;
    int sock_fd_;
    void *conf_space_;
    std::vector<Device *> devices_;
};

class EnclaveGuestDevice : public Device {
  public:
    friend class EnclaveGuestPlatform;
    friend class GuestRPCClient;
    virtual unsigned GetNodeID() const override { return node_id_; }
    virtual unsigned GetGPUID() const override { return gpu_id_; }
    virtual unsigned GetDoorbellPageSize() const override { return 8192; }
    virtual void *
    GetOrInitializeDoorbell(uint64_t doorbell_mmap_offset) override {
        HSA_ASSERT(0 && "Unimplemented");
    }
    virtual const Properties &GetProperties() const override;
    virtual MemoryManager *GetMemoryManager() override { return mm_.get(); }
    virtual std::unique_ptr<DeviceQueue> CreateSDMAQueue() override;
    virtual std::unique_ptr<DeviceQueue> CreateAQLQueue() override;

    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;

    GuestRPCClient *GetRPCClient() const { return client_.get(); }

  private:
    explicit EnclaveGuestDevice(unsigned node_id, unsigned gpu_id,
                                TransmitBuffer &&tx, TransmitBuffer &&rx);
    void NotifyHostAgent();
    unsigned node_id_;
    unsigned gpu_id_;
    TransmitBuffer tx_;
    TransmitBuffer rx_;
    std::unique_ptr<MemoryManager> mm_;
    std::unique_ptr<GuestRPCClient> client_;
};

} // namespace ocl::hsa::enclave
