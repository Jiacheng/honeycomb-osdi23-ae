#pragma once

#include <absl/status/status.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <memory>

namespace gpumpc::rpc {

class IPCSharedMemory {
  public:
    virtual ~IPCSharedMemory() = default;
    virtual absl::Status Close() = 0;
    virtual void *GetBuffer() = 0;
    virtual const void *GetHandle() const = 0;
};

template <class T> class IPCSharedMemoryFactory {
  public:
    typedef T Handle;
    virtual ~IPCSharedMemoryFactory() = default;
    virtual absl::Status
    CreateSharedMemory(size_t size, std::unique_ptr<IPCSharedMemory> *res) = 0;
    virtual absl::Status
    AttachSharedMemory(const Handle &handle,
                       std::unique_ptr<IPCSharedMemory> *res) = 0;
};

std::unique_ptr<IPCSharedMemoryFactory<hipIpcMemHandle_t>>
CreateHIPSharedMemoryFactory();
} // namespace gpumpc::rpc