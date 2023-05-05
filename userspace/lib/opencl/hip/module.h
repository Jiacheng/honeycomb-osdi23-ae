#pragma once

#include "elf/amdgpu_program.h"
#include "opencl/hsa/memory_manager.h"
#include "opencl/hsa/types.h"
#include <absl/status/status.h>
#include <map>
#include <memory>
#include <string_view>

namespace ocl::hip {

class DeviceContext;

class Module;
class Kernel {
  public:
    friend class Module;
    Module *GetParent() const { return parent_; }
    const AMDGPUProgram::KernelInfo *GetKernelInfo() const { return ki_; }

  private:
    friend class Module;
    explicit Kernel(Module *parent, const AMDGPUProgram::KernelInfo *ki);
    Kernel(const Kernel &) = delete;
    Kernel &operator=(const Kernel &) = delete;

    Module *parent_;
    const AMDGPUProgram::KernelInfo *ki_;
};

class Module {
  public:
    ~Module();

    //
    // Initialize a module and load it into the device memory.
    // It is done at the same time to avoid extra copy
    static absl::Status New(DeviceContext *parent, const char *data,
                            std::unique_ptr<Module> *ret);

    const Kernel *GetKernel(std::string_view name) const {
        auto it = kernels_.find(std::string(name));
        return it != kernels_.end() ? it->second.get() : nullptr;
    }

    ocl::hsa::gpu_addr_t GetVMABase() const { return d_blob_; }

  private:
    explicit Module(DeviceContext *ctx);
    Module(const Module &) = delete;
    Module &operator=(const Module &) = delete;

    //
    // Allocate the memory for the ELF and load it into the device memory.
    absl::Status Load(std::string_view blob);
    absl::Status PopulateKernels();

    DeviceContext *parent_;

    std::unique_ptr<AMDGPUProgram> prog_;
    std::map<std::string, std::unique_ptr<Kernel>> kernels_;

    //
    // The ELF image on the device memory
    // TODO: Use a custom allocator to put the blob in the right place
    ocl::hsa::gpu_addr_t d_blob_;
};

} // namespace ocl::hip