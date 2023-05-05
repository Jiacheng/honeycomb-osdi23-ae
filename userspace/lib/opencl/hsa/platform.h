#pragma once

#include "event.h"
#include "memory_manager.h"

#include <absl/status/status.h>
#include <memory>
#include <vector>

namespace ocl::hsa {

class Platform;
class KFDPlatform;
class Event;
class DeviceQueue;

class Device {
  public:
    enum {
        kPageSize = 4096,
        kHugeGPUPageSize = 2 * 1024 * 1024,
    };
    struct Properties {
        unsigned num_fcompute_cores;
        unsigned num_simd_per_cu;
        unsigned num_waves;
        unsigned control_stack_bytes_per_wave;
        unsigned wg_context_data_size_per_cu;
        unsigned eop_buffer_size;
        unsigned debugger_bytes_per_wave;
        unsigned debugger_bytes_align;
    };

    friend class Platform;
    friend class KFDPlatform;
    virtual unsigned GetNodeID() const = 0;
    virtual unsigned GetGPUID() const = 0;
    virtual unsigned GetDoorbellPageSize() const = 0;
    virtual void *GetOrInitializeDoorbell(uint64_t doorbell_mmap_offset) = 0;
    virtual const Properties &GetProperties() const = 0;
    virtual MemoryManager *GetMemoryManager() = 0;
    virtual std::unique_ptr<DeviceQueue> CreateSDMAQueue() = 0;
    virtual std::unique_ptr<DeviceQueue> CreateAQLQueue() = 0;

    virtual absl::Status Initialize() = 0;
    virtual absl::Status Close() = 0;

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    virtual ~Device() = default;

  protected:
    Device() = default;
};

class Platform {
  public:
    enum Variant {
        kPlatformKFD,
        kPlatformG6,
        kPlatformEnclaveGuest,
    };
    enum { kEventSlotSizeBytes = sizeof(uint64_t) };

    virtual ~Platform() = default;
    static void ChooseVariant(Variant variant);
    static Platform &Instance();

    virtual int GetKFDFD() const = 0;
    virtual const std::vector<Device *> &GetDevices() const = 0;
    virtual gpu_addr_t GetEventPageBase() = 0;
    virtual absl::Status Initialize() = 0;
    virtual absl::Status Close() = 0;
    virtual std::unique_ptr<Event> NewEvent(int type,
                                            uint64_t event_page_handle) = 0;
    virtual std::unique_ptr<Event> NewSignalEvent();

  protected:
    Platform() = default;
    Platform(const Platform &) = delete;
    Platform &operator=(const Platform &) = delete;

    static Variant variant_;
};

} // namespace ocl::hsa