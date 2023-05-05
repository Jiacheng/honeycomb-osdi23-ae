#pragma once

#include "opencl/hsa/types.h"

#include <absl/status/status.h>

#include <map>
#include <memory>
#include <string>

namespace ocl::hip {

// Arguments of the kernels. The offset starts with sizeof(ImplicitArgument).
class KernelArgument {
  public:
#pragma pack(push, 1)
    struct ImplicitArgumentLLVM {
        uint64_t global_offset[3];
        ocl::hsa::gpu_addr_t printf_buffer;
        ocl::hsa::gpu_addr_t vqueue;
        ocl::hsa::gpu_addr_t aqlwrap;
        ocl::hsa::gpu_addr_t multigrid_sync_arg;
    };
#pragma pack(pop)
    static const int kImplicitArgumentNumLLVM = 7;

    enum {
        kArgPointer,
        kArgPrimitive,
    };
    explicit KernelArgument(const std::string &name, unsigned type, size_t size,
                            size_t offset);
    const std::string &GetName() const { return name_; }
    unsigned GetType() const { return type_; }
    size_t GetSize() const { return size_; }
    size_t GetOffset() const { return offset_; }
    static KernelArgument Invalid() { return KernelArgument("", 0, 0, 0); }

  private:
    std::string name_;
    unsigned type_;
    size_t size_;
    size_t offset_;
};

class AMDGPUProgram {
  public:
    friend class LLVMELFParser;
    friend class ELFParserBase;
    struct KernelInfo {
        // The relative VMA that points to the corresponding kernel descriptor
        uint64_t desc_vma_offset;
        // kernarg_segment_size
        size_t kernarg_size;
        // kernarg_segment_align
        size_t kernarg_align;
        // group_segment_fixed_size
        unsigned lds_size;
        unsigned private_segment_fixed_size;
        std::vector<KernelArgument> args;
        KernelInfo() = default;
    };

    struct Segment {
        size_t file_offset;
        size_t file_size;
        size_t vma_start;
        size_t vma_length;
    };

    const std::vector<Segment> &GetLoadSegment() const {
        return load_segments_;
    }

    const std::map<std::string, KernelInfo> &GetKernels() const {
        return kernels_;
    }

    size_t GetVMAEnd() const;

  private:
    std::vector<Segment> load_segments_;
    std::map<std::string, KernelInfo> kernels_;
};

std::unique_ptr<AMDGPUProgram> ParseAMDGPUProgram(std::string_view blob,
                                                  absl::Status *ret);
} // namespace ocl::hip
