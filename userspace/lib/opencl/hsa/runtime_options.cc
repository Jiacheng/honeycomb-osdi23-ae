#include "runtime_options.h"
#include "opencl/hsa/assert.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <climits>
#include <cstdlib>

namespace ocl {

RuntimeOptions::RuntimeOptions() {
    auto stat = Initialize();
    initialized_ = stat.ok();
}

static absl::Status ParseInt(long *target, const char *env_var) {
    auto ret = getenv(env_var);
    if (!ret) {
        *target = 0;
        return absl::OkStatus();
    }
    *target = strtol(ret, nullptr, 10);
    return absl::OkStatus();
}

static absl::Status ParseBool(bool *target, const char *env_var) {
    long v = 0;
    auto stat = ParseInt(&v, env_var);
    if (!stat.ok()) {
        return stat;
    } else if (v == LONG_MAX || v == LONG_MIN) {
        return absl::InvalidArgumentError(std::string(env_var) +
                                          " is not properly set");
    }
    *target = v;
    return absl::OkStatus();
}

static absl::Status ParseEnvString(std::string *target, const char *env_var) {
    auto ret = getenv(env_var);
    if (!ret) {
        return absl::InvalidArgumentError(std::string(env_var) +
                                          " must be a string");
    }
    *target = ret;
    return absl::OkStatus();
}

absl::Status RuntimeOptions::Initialize() {
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner
        .Run([&]() {
            return ParseBool(&secure_memcpy_, "GPUMPC_SECURE_MEMCPY");
        })
        .Run([&]() {
            return ParseBool(&strict_layout_, "GPUMPC_STRICT_LAYOUT");
        })
        .Run([&]() {
            if (secure_memcpy_) {
                return ParseEnvString(&resource_dir_, "GPUMPC_RESOURCE_DIR");
            } else {
                return absl::OkStatus();
            }
        })
        .Run([&]() {
            long variant = 0;
            auto stat = ParseInt(&variant, "GPUMPC_ENCLAVE");
            if (!stat.ok()) {
                return stat;
            }
            enclave_ = (EnclaveVariant)variant;
            return absl::OkStatus();
        })
        .Run([&]() {
            if (enclave_) {
                return ParseEnvString(&enclave_shm_, "GPUMPC_ENCLAVE_SHM");
            } else {
                return absl::OkStatus();
            }
        })
        .Run([&]() {
            if (enclave_ == kEnclaveFull) {
                return ParseEnvString(&agent_physical_memory_,
                                      "GPUMPC_ENCLAVE_AGENT_MEM");
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
