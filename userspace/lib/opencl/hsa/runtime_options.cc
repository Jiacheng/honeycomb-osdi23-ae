#include "runtime_options.h"
#include "opencl/hsa/assert.h"
#include "utils/monad_runner.h"

namespace ocl {

RuntimeOptions::RuntimeOptions() {
    auto stat = Initialize();
    initialized_ = stat.ok();
}

absl::Status RuntimeOptions::ParseEnv(bool *target,
                                      const std::string &env_var) {
    auto ret = getenv(env_var.c_str());
    if (!ret) {
        *target = false;
        return absl::OkStatus();
    }
    std::string value = ret;
    std::string one = "1";
    std::string zero = "0";
    if (value == zero) {
        *target = false;
    } else if (value == one) {
        *target = true;
    } else {
        return absl::InvalidArgumentError(env_var +
                                          " is not properly set: " + value);
    }
    return absl::OkStatus();
}

absl::Status RuntimeOptions::ParseEnvString(std::string *target,
                                            const std::string &env_var) {
    auto ret = getenv(env_var.c_str());
    if (!ret) {
        HSA_ASSERT(0 && "Must set env");
    }
    *target = ret;
    return absl::OkStatus();
}

absl::Status RuntimeOptions::Initialize() {
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run(
            [&]() { return ParseEnv(&secure_memcpy_, "GPUMPC_SECURE_MEMCPY"); })
        .Run(
            [&]() { return ParseEnv(&strict_layout_, "GPUMPC_STRICT_LAYOUT"); })
        .Run([&]() {
            if (secure_memcpy_) {
                return ParseEnvString(&resource_dir_, "GPUMPC_RESOURCE_DIR");
            } else {
                return absl::OkStatus();
            }
        })
        .Run([&]() { return ParseEnv(&enclave_, "GPUMPC_ENCLAVE"); })
        .Run([&]() {
            if (enclave_) {
                return ParseEnvString(&enclave_shm_, "GPUMPC_ENCLAVE_SHM");
            } else {
                return absl::OkStatus();
            }
        })
        .Run([&]() {
            if (enclave_) {
                return ParseEnvString(&enclave_socket_,
                                      "GPUMPC_ENCLAVE_SOCKET");
            } else {
                return absl::OkStatus();
            }
        });
    return runner.code();
}

RuntimeOptions *GetRuntimeOptions() {
    static RuntimeOptions opt;
    return opt.IsInitialized() ? &opt : nullptr;
}

} // namespace ocl
