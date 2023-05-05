#include "experiments/platform.h"
#include <absl/status/status.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace gpumpc::experiment {
namespace fs = std::filesystem;

class LinuxPlatform : public ExperimentPlatform {
  public:
    virtual absl::Status Initialize() override;
    virtual absl::Status Close() override;
    //
    // Load resource (e.g., binary data) from the OS
    virtual absl::Status LoadResource(const std::string &name,
                                      std::vector<char> *result) override;
    fs::path base_path_;
};

absl::Status LinuxPlatform::Initialize() {
    auto resource_dir = std::getenv("GPUMPC_RESOURCE_DIR");
    if (!resource_dir) {
        return absl::InvalidArgumentError("resource directory not found");
    }
    base_path_ = resource_dir;
    return absl::OkStatus();
}

absl::Status LinuxPlatform::Close() { return absl::OkStatus(); }

//
// Load resource (e.g., binary data) from the OS
absl::Status LinuxPlatform::LoadResource(const std::string &name,
                                         std::vector<char> *result) {
    std::error_code ec;
    auto path = base_path_ / name;
    auto size = file_size(path, ec);
    if (ec) {
        return absl::InvalidArgumentError("Cannot open file");
    }
    result->resize(size);

    std::ifstream is(path, std::ifstream::binary);
    is.read(result->data(), size);
    if (is.gcount() != (long)size) {
        return absl::InvalidArgumentError("Premature EOF");
    }
    return absl::OkStatus();
}

ExperimentPlatform &ExperimentPlatform::Instance() {
    static LinuxPlatform inst;
    return inst;
}

} // namespace gpumpc::experiment