#pragma once

#include "amdgpu_program.h"
#include <absl/status/status.h>

#include <map>
#include <string_view>

namespace ocl::hip::msgpack {
class Map;
}

namespace ocl::hip {

class CodeObjectV3MetadataParser {
  public:
    explicit CodeObjectV3MetadataParser(
        const std::map<std::string, uint64_t> &kd_vmas);
    absl::Status Parse(std::string_view data,
                       std::map<std::string, AMDGPUProgram::KernelInfo> *out);

  private:
    absl::Status ParseKernelInfo(const msgpack::Map *m, std::string *name,
                                 AMDGPUProgram::KernelInfo *ki);
    KernelArgument ParseArgument(msgpack::Map *m, absl::Status *stat);
    const std::map<std::string, uint64_t> &kd_vmas_;
};
} // namespace ocl::hip
