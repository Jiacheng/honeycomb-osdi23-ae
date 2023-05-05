#pragma once
#include "opencl/hip/usm/secure_memcpy.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <gflags/gflags.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <openssl/evp.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace gpumpc::experiment {

class CPURPC {
  public:
    enum Mode {
        kServer,
        kClient,
    };
    struct Options {
        std::string module_path;
        std::string ipc_mem_file;
        Mode mode;
    };

    virtual absl::Status Initialize(const Options &options);
    virtual absl::Status Destroy();
    absl::Status CalibrateTime();
    absl::Status Launch();

    ~CPURPC();

  protected:
    absl::Status InitializeServer();
    absl::Status InitializeClient();
    virtual absl::Status LaunchServer() = 0;
    virtual absl::Status LaunchClient() = 0;

    absl::Status GetTimestamp();
    absl::Status FetchRequest();
    absl::Status PutResponse();

    absl::Status LoadModule(hipModule_t *module, std::string module_path);
    absl::Status LoadBinary();

    void Signal(std::atomic_ulong *direction);
    void Wait(std::atomic_ulong *direction);

    absl::Status EncryptToPayload();
    absl::Status DecryptFromPayload();

    enum {
        kDeviceBaseSize = 2 << 20, // also for calibrate
        kClientSignalOffset = 0,
        kServerSignalOffset = sizeof(unsigned long),
        kTimestampOffset = 2 * sizeof(unsigned long),
        kPayloadOffset = 3 * sizeof(unsigned long),
    };

    // should be set in the child's constructor
    size_t payload_size_;

    Options options_;
    hipModule_t module_;
    hipFunction_t server_;
    hipFunction_t client_;
    hipFunction_t client_collect_;
    hipFunction_t get_clock_;
    hipDeviceptr_t base_;
    hipDeviceptr_t request_;
    double ns_per_tsc_;
    unsigned long timestamp_;
    std::vector<char> buf_;
    int shm_fd_;
    unsigned char *shm_;
    size_t shm_size_;
    std::atomic_ulong *client_signal_;
    std::atomic_ulong *server_signal_;
    unsigned char *timestamp_shm_;
    unsigned char *payload_;
    EVP_CIPHER_CTX *ctx_;
    unsigned ukey_[ocl::hip::kAES256KeySizeInWord];
    unsigned iv_[ocl::hip::kAESIVSizeInWord];
};

} // namespace gpumpc::experiment
