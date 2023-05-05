#pragma once

#include "rpc/ipc_shm.h"
#include "rpc/ring_queue.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <absl/status/status.h>

namespace gpumpc::experiment {

class GPUDirectRPC {
  public:
    enum Variant {
        kRawRPCPerformance,
        kRPCWithCapability,
    };
    enum Mode {
        kServer,
        kClient,
    };
    struct Options {
        Variant variant;
        std::string module_path; 
        std::string ipc_mem_handle_file;
        Mode mode;
        unsigned block_size;
    };

    enum {
        kSharedMemorySize = 2 << 20,
    };

    virtual absl::Status Initialize(const Options &options);
    absl::Status CalibrateTime();
    absl::Status Launch();
    virtual ~GPUDirectRPC() = default;

protected:
    absl::Status InitializeServer(hipIpcMemHandle_t *handle);
    absl::Status InitializeClient(hipIpcMemHandle_t handle);
    void InitializeQueueView();
    virtual absl::Status LaunchServer();
    virtual absl::Status LaunchClient();

    absl::Status LoadModule(hipModule_t *module, const std::string &module_path);
    absl::Status LoadBinary();

    Options options_;
    hipModule_t module_;
    hipFunction_t server_;
    hipFunction_t client_;
    hipFunction_t get_clock_;
    std::unique_ptr<rpc::IPCSharedMemoryFactory<hipIpcMemHandle_t>> factory_;
    std::unique_ptr<rpc::IPCSharedMemory> mem_;
    double ns_per_tsc_;
    rpc::LockFreeQueueView queue_;
};

} // namespace gpumpc::experiment