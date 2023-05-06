#include "cl_program.h"
#include "absl/strings/escaping.h"
#include "cl_context.h"
#include "crypto/sha3.h"
#include "utils/filesystem.h"

#include <filesystem>
#include <iomanip>

namespace crater::opencl {

Program::Program(Context *ctx, hipModule_t module)
    : ctx_(ctx), module_(module) {}

Program::~Program() {
    auto _ = ctx_->GetCtx()->GetComputeContext()->hipModuleUnload(module_);
    (void)_;
    ctx_->Release();
}

cl_program Program::clCreateProgramWithSource(cl_context context, cl_uint count,
                                              const char **strings,
                                              const size_t *lengths,
                                              cl_int *errcode_ret) {
    namespace fs = std::filesystem;
    if (errcode_ret) {
        *errcode_ret = CL_INVALID_VALUE;
    }

    const char *cache_dir = std::getenv("CL_CACHE_DIR");
    if (!cache_dir) {
        return nullptr;
    }

    auto binary_home = fs::path(std::getenv("CL_CACHE_DIR"));

    // concatence input strings into one
    std::string program_string;
    for (size_t i = 0; i < count; i++) {
        if (lengths != NULL) {
            program_string.append(std::string(strings[i], lengths[i]));
        } else {
            program_string.append(std::string(strings[i]));
        }
    }

    char output[32];
    SHA3_256(reinterpret_cast<unsigned char *>(output),
             reinterpret_cast<unsigned char *>(program_string.data()),
             program_string.length());
    std::string hex_hash_value =
        absl::BytesToHexString(absl::string_view(output, 32));
    auto file_name = fs::path(std::string("cl-") + hex_hash_value + ".bin");
    auto binary_path = binary_home / file_name;

    absl::Status stat;
    auto binary = gpumpc::ReadAll(binary_path, &stat);
    if (stat.ok() && !binary.empty()) {
        size_t binary_size = binary.size();
        auto ctx = static_cast<Context *>(context);
        cl_device_id device = ctx->GetDevices()[0];
        auto binary_data_pointer =
            reinterpret_cast<const unsigned char *>(binary.data());

        return Program::clCreateProgramWithBinary(
            context, 1, &device, &binary_size, &binary_data_pointer, NULL,
            errcode_ret);
    }
    return nullptr;
}

cl_program Program::clCreateProgramWithBinary(
    cl_context context, cl_uint num_devices, const cl_device_id *device_list,
    const size_t *lengths, const unsigned char **binaries,
    cl_int *binary_status, cl_int *errcode_ret) {
    auto ctx = static_cast<Context *>(context);
    auto hip = ctx->GetCtx();

    // Support a single device right now
    if (num_devices != 1) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }

    std::string_view blob(reinterpret_cast<const char *>(binaries[0]),
                          lengths[0]);

    hipModule_t module;
    auto error =
        hip->GetComputeContext()->hipModuleLoadData(&module, blob.data());

    if (error != hipSuccess) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_VALUE;
        }
        return nullptr;
    }
    if (errcode_ret) {
        *errcode_ret = CL_SUCCESS;
    }
    ctx->Retain();
    return new Program(ctx, module);
}

cl_int Program::clBuildProgram(cl_program program, cl_uint num_devices,
                               const cl_device_id *device_list,
                               const char *options,
                               void(CL_CALLBACK *pfn_notify)(cl_program program,
                                                             void *user_data),
                               void *user_data) {
    return CL_SUCCESS;
}

cl_int Program::clRetainProgram(cl_program program) {
    auto self = static_cast<Program *>(program);
    self->Retain();
    return CL_SUCCESS;
}

cl_int Program::clReleaseProgram(cl_program program) {
    auto self = static_cast<Program *>(program);
    self->Release();
    return CL_SUCCESS;
}

cl_int Program::clGetProgramBuildInfo(cl_program program, cl_device_id device,
                                      cl_program_build_info param_name,
                                      size_t param_value_size,
                                      void *param_value,
                                      size_t *param_value_size_ret) {
    if (param_value_size_ret != NULL)
        *param_value_size_ret = 0;
    return CL_SUCCESS;
}

} // namespace crater::opencl
