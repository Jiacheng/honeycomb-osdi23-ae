#pragma once

#include <absl/status/status.h>
#include <absl/types/span.h>

typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;

namespace gpumpc::experiment {

class ResNetInference {
  public:
    enum { kImageSize = 294912, kResultSize = 256 << 10 };
    static constexpr char kDataPrefix[] = "data/resnet";

    virtual ~ResNetInference() = default;
    virtual absl::Status Initialize() = 0;
    virtual absl::Status Close() = 0;

    //
    // Convienent API to infer the results for a single image.
    // TODO: Need to add more APIs for evaluations.
    virtual absl::Status Run(absl::Span<const char> image) = 0;

    // HACK: Run the pipeline without copying
    virtual absl::Status RunDirect() { return absl::OkStatus(); }

    //
    // Synchronize and copy the result of inference to the host
    // memory.
    // It expects the size of the result span equals to kResultSize.
    virtual absl::Status Fetch(absl::Span<char> result) = 0;

    //
    // HACK: Return the address that is used in Run() to store the image
    virtual void *GetImageSourceBufferAddr() { return nullptr; }
    //
    // HACK: Return the address that is used in Fetch() to store the result
    virtual void *GetResultBufferAddr() { return nullptr; }

    ResNetInference(const ResNetInference &) = delete;
    ResNetInference &operator=(const ResNetInference &) = delete;

  protected:
    explicit ResNetInference() = default;

    struct FunctionDescriptor {
      const char *name;
      unsigned module_index;
    };
};

class ResNetInferenceImpl : public ResNetInference {
  protected:
    absl::Status LoadFunctions(
      absl::Span<std::string> module_name,
      absl::Span<const FunctionDescriptor> func_desc
    );
    absl::Status DestroyFunctions();
    std::vector<hipFunction_t> func_;
    std::vector<hipModule_t> module_;
};

std::unique_ptr<ResNetInference> NewResNet1();
std::unique_ptr<ResNetInference> NewResNet1Baseline();
std::unique_ptr<ResNetInference> NewResNet18();
} // namespace gpumpc::experiment
