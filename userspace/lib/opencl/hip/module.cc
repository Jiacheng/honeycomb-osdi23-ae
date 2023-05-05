#include "module.h"
#include "device_context.h"
#include "elf/parser.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <memory>
#include <string_view>

namespace ocl::hip {

Module::Module(DeviceContext *ctx) : parent_(ctx), d_blob_(0) {}

Module::~Module() {
    auto _ = parent_->GetMemoryManager()->hipFree(d_blob_);
    (void)_;
}

absl::Status Module::New(DeviceContext *parent, const char *data,
                         std::unique_ptr<Module> *ret) {
    absl::Status stat;
    auto length = ELFParserBase::GuessELFBinarySize(data);
    auto elf = std::string_view(data, length);
    std::unique_ptr<Module> mod(new Module(parent));

    gpumpc::MonadRunner<absl::Status> r(absl::OkStatus());
    r.Run([&]() {
         mod->prog_ = ParseAMDGPUProgram(elf, &stat);
         return stat;
     })
        .Run([&]() { return mod->Load(elf); })
        .Run([&]() { return mod->PopulateKernels(); });

    if (!r.code().ok()) {
        return r.code();
    }
    *ret = std::move(mod);
    return absl::OkStatus();
}

//
// TODO: Use a custom allocator to control the address space and reduce
// fragment
absl::Status Module::Load(std::string_view blob) {
    auto mm = parent_->GetMemoryManager();

    auto r = mm->hipMalloc(&d_blob_, prog_->GetVMAEnd());
    if (r != hipSuccess) {
        return absl::ResourceExhaustedError(
            "Cannot allocate memory for the module");
    }

    for (const auto &s : prog_->GetLoadSegment()) {
        auto f = blob.substr(s.file_offset, s.file_size);
        //
        // .bss has a zero file size but positive vma length.
        // Add the support later on
        if (f.size() != s.vma_length) {
            return absl::InvalidArgumentError("Unimplemented");
        }
        auto r = mm->hipMemcpyHtoD(d_blob_ + s.vma_start, f.data(), f.size());
        if (r != hipSuccess) {
            return absl::ResourceExhaustedError(
                "Cannot copy the code to the device memory");
        }
    }

    return absl::OkStatus();
}

absl::Status Module::PopulateKernels() {
    for (auto &k : prog_->GetKernels()) {
        auto v = std::unique_ptr<Kernel>(new Kernel(this, &k.second));
        kernels_.insert({k.first, std::move(v)});
    }
    return absl::OkStatus();
}

Kernel::Kernel(Module *parent, const AMDGPUProgram::KernelInfo *ki)
    : parent_(parent), ki_(ki) {}

} // namespace ocl::hip
