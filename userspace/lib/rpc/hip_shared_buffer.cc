#include "ipc_shm.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <memory>

namespace gpumpc::rpc {

class HIPIPCSharedMemory : public IPCSharedMemory {
  public:
    enum Mode {
        kServer,
        kClient,
    };
    explicit HIPIPCSharedMemory(Mode mode, void *mem, hipIpcMemHandle_t handle)
        : mode_(mode), mem_(mem), handle_(handle) {}
    virtual ~HIPIPCSharedMemory();
    virtual absl::Status Close() override;
    virtual void *GetBuffer() override { return mem_; }
    virtual const void *GetHandle() const override {
        return reinterpret_cast<const void *>(&handle_);
    }

    Mode mode_;
    void *mem_;
    hipIpcMemHandle_t handle_;
};

class HIPIPCSharedMemoryFactory
    : public IPCSharedMemoryFactory<hipIpcMemHandle_t> {
  public:
    virtual absl::Status
    CreateSharedMemory(size_t size,
                       std::unique_ptr<IPCSharedMemory> *res) override;
    virtual absl::Status
    AttachSharedMemory(const Handle &handle,
                       std::unique_ptr<IPCSharedMemory> *res) override;
};

absl::Status HIPIPCSharedMemoryFactory::CreateSharedMemory(
    size_t size, std::unique_ptr<IPCSharedMemory> *res) {
    void *memory = nullptr;
    hipIpcMemHandle_t handle;

    MonadRunner<hipError_t> runner(hipSuccess);
    runner.Run([&]() { return hipMalloc(&memory, size); }).Run([&]() {
        return hipIpcGetMemHandle(&handle, memory);
    });
    if (runner.code()) {
        if (memory) {
            auto _ = hipFree(memory);
            (void)_;
            return absl::InvalidArgumentError(
                "Cannot create shared HIP shared memory");
        }
    }
    res->reset(
        new HIPIPCSharedMemory(HIPIPCSharedMemory::kServer, memory, handle));
    return absl::OkStatus();
}

absl::Status HIPIPCSharedMemoryFactory::AttachSharedMemory(
    const Handle &handle, std::unique_ptr<IPCSharedMemory> *res) {
    void *memory = nullptr;
    if (hipIpcOpenMemHandle(&memory, handle, 0)) {
        return absl::InvalidArgumentError(
            "Cannot attach the HIP shared memory");
    }
    res->reset(
        new HIPIPCSharedMemory(HIPIPCSharedMemory::kClient, memory, handle));
    return absl::OkStatus();
}
HIPIPCSharedMemory::~HIPIPCSharedMemory() {
    if (mem_) {
        auto _ = Close();
        (void)_;
    }
}

absl::Status HIPIPCSharedMemory::Close() {
    if (mode_ == kServer) {
        auto _ = hipIpcCloseMemHandle(mem_);
        (void)_;
    }

    auto err = hipFree(mem_);
    if (err) {
        return absl::InvalidArgumentError("Cannot deallocate the memory");
    }
    return absl::OkStatus();
}

std::unique_ptr<IPCSharedMemoryFactory<hipIpcMemHandle_t>>
CreateHIPSharedMemoryFactory() {
    return std::unique_ptr<IPCSharedMemoryFactory<hipIpcMemHandle_t>>(
        new HIPIPCSharedMemoryFactory());
}

} // namespace gpumpc::rpc