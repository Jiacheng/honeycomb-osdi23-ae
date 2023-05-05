#pragma once

#include <absl/status/status.h>
#include <string>

namespace ocl {

class RuntimeOptions;

RuntimeOptions *GetRuntimeOptions();

class RuntimeOptions {
  public:
    friend RuntimeOptions *GetRuntimeOptions();
    bool IsInitialized() const { return initialized_; }
    bool IsSecureMemcpy() const { return secure_memcpy_; }
    bool IsStrictLayout() const { return strict_layout_; }
    bool IsEnclave() const { return enclave_; }
    const std::string &GetResourceDir() const { return resource_dir_; }
    const std::string &GetEnclaveShm() const { return enclave_shm_; }
    const std::string &GetEnclaveSocket() const { return enclave_socket_; }

  private:
    RuntimeOptions();
    absl::Status Initialize();
    absl::Status ParseEnv(bool *target, const std::string &env_var);
    absl::Status ParseEnvString(std::string *target,
                                const std::string &env_var);

    bool initialized_;

    bool secure_memcpy_;
    bool strict_layout_;
    bool enclave_;

    std::string resource_dir_;
    std::string enclave_shm_;
    std::string enclave_socket_;
};

} // namespace ocl
