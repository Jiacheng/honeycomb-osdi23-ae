#include "experiments/resnet/resnet_inference.h"
#include "experiments/platform.h"
#include <absl/status/status.h>
#include <hip/hip_runtime_api.h>

namespace gpumpc::experiment {
absl::Status
ResNetInferenceImpl::LoadFunctions(absl::Span<std::string> module_path,
                               absl::Span<const FunctionDescriptor> func_desc) {
    module_.resize(module_path.size());
    auto &plat = ExperimentPlatform::Instance();
    for (size_t i = 0; i < module_path.size(); i++) {
        std::vector<char> data;
        auto stat = plat.LoadResource(module_path[i], &data);
        if (!stat.ok()) {
            return stat;
        }
        auto err = hipModuleLoadData(&module_[i], data.data());
        if (err != hipSuccess) {
            return absl::InvalidArgumentError("Cannot load kernel");
        }
    }

    func_.resize(func_desc.size());
    for (size_t i = 0; i < func_desc.size(); i++) {
        auto err = hipModuleGetFunction(&func_[i],
                                        module_.at(func_desc[i].module_index),
                                        func_desc[i].name);
        if (err != hipSuccess) {
            return absl::InvalidArgumentError("Cannot load function");
        }
    }
    return absl::OkStatus();
}

absl::Status ResNetInferenceImpl::DestroyFunctions() {
    for (const auto &m : module_) {
        auto _ = hipModuleUnload(m);
        (void)_;
    }
    return absl::OkStatus();
}

} // namespace gpumpc::experiment
