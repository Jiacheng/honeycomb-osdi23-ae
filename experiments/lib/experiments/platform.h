#pragma once

#include <absl/status/status.h>
#include <vector>

namespace gpumpc::experiment {

//
// The experiment platform provides basic abstractions to interact with the OS.
class ExperimentPlatform {
public:
    static ExperimentPlatform &Instance(); 
    virtual ~ExperimentPlatform() = default;
    virtual absl::Status Initialize() = 0;
    virtual absl::Status Close() = 0;
    //
    // Load resource (e.g., binary data) from the OS. For simplicity the function
    // loads the full content of the resource 
    virtual absl::Status LoadResource(const std::string &name, std::vector<char> *result) = 0;

    ExperimentPlatform(const ExperimentPlatform &) = delete;
    ExperimentPlatform &operator=(const ExperimentPlatform &) = delete;
protected:
    explicit ExperimentPlatform() = default;
};

}