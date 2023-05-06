#pragma once

#include <absl/status/status.h>
#include <string>

namespace ocl {

class RuntimeOptions;

RuntimeOptions *GetRuntimeOptions();

class RuntimeOptions {
  public:
    enum EnclaveVariant {
        kEnclaveNone,
        kEnclaveLegacy,
        kEnclaveFull,
    };
    friend RuntimeOptions *GetRuntimeOptions();
    bool IsInitialized() const { return initialized_; }
    bool IsSecureMemcpy() const { return secure_memcpy_; }
    bool IsStrictLayout() const { return strict_layout_; }
    EnclaveVariant GetEnclaveVariant() const { return enclave_; }
    bool MapRemotePhysicalPage() const { return enclave_ == EnclaveVariant::kEnclaveFull; }
    const std::string &GetResourceDir() const { return resource_dir_; }
    const std::string &GetEnclaveShm() const { return enclave_shm_; }
    const std::string &GetAgentPhysicalMemoryPath() const {
        return agent_physical_memory_;
    }

  private:
    RuntimeOptions();
    absl::Status Initialize();

    bool initialized_;

    bool secure_memcpy_;
    bool strict_layout_;
    EnclaveVariant enclave_;

    std::string resource_dir_;
    std::string enclave_shm_;
    std::string agent_physical_memory_;
};

} // namespace ocl
