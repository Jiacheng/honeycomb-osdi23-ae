#include "opencl_test_utils.h"
#include <absl/status/status.h>

#include <fstream>
#include <vector>

namespace crater::opencl {

#define checkCLErrors(call)                                                    \
    do {                                                                       \
        call;                                                                  \
        if (err) {                                                             \
            std::cerr << "OpenCL Error '" << err << "' in func '"              \
                      << __FUNCTION__ << "' line " << __LINE__;                \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

using absl::Status;

static std::vector<char> ReadAll(const std::string &fn, Status *stat) {
    std::ifstream is(fn, std::ifstream::binary);
    std::vector<char> result;
    char buf[4096];
    while (!is.bad() && !is.eof()) {
        is.read(buf, sizeof(buf));
        size_t count = is.gcount();
        if (!count) {
            break;
        }
        if (!is.bad()) {
            result.insert(result.end(), buf, buf + count);
        }
    }
    if (is.bad()) {
        *stat = absl::InvalidArgumentError("Failed to read the file " + fn);
        return std::vector<char>();
    }
    *stat = absl::OkStatus();
    return result;
}

static const char kAMDPlatformName[] = "AMD Accelerated Parallel Processing";

std::vector<cl::Device> EnumerateAMDGPUs() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform;

    for (auto p : platforms) {
        auto r = p.getInfo<CL_PLATFORM_NAME>();
        if (!strncmp(r.c_str(), kAMDPlatformName,
                     sizeof(kAMDPlatformName) - 1)) {
            platform = p;
            break;
        }
    }

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    return devices;
}

void BareMetalOpenCLTest::SetUp() {
    const std::vector<cl::Device> devices = EnumerateAMDGPUs();
    size_t device_idx = 0;
    for (; device_idx < devices.size(); ++device_idx) {
		const std::string device_name = devices[device_idx].getInfo<
				CL_DEVICE_NAME>();
    	if (std::strcmp(device_name.c_str(), "gfx1030") == 0) {
    		break;
    	}
    }
    ASSERT_LT(device_idx, devices.size());
    device_ = devices[device_idx];
    auto platform = cl::Platform(device_.getInfo<CL_DEVICE_PLATFORM>());

    cl_int err = 0;
    cl_context_properties ctx_prop[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)(platform()), 0};
    checkCLErrors(ctx_ =
                      cl::Context(device_, ctx_prop, nullptr, nullptr, &err));
    checkCLErrors(queue_ = cl::CommandQueue(ctx_, device_, 0, &err));
}

void BareMetalOpenCLTest::MustLoadBinaryFromFile(const std::string &filename,
                                                 cl::Program *prog) {
    Status stat;
    std::vector<char> kernel_blob = ReadAll(filename, &stat);
    ASSERT_TRUE(stat.ok());
    cl_int err = 0;

    cl::Program::Binaries binaries;
    binaries.push_back(std::make_pair(kernel_blob.data(), kernel_blob.size()));

    std::vector<cl::Device> devices{device_};
    *prog = std::move(cl::Program(ctx_, devices, binaries, nullptr, &err));
    ASSERT_EQ(0, err);
    err = prog->build(devices);
    ASSERT_EQ(0, err);
}

void BareMetalOpenCLTest::MustLoadKernel(const cl::Program &program,
                                         const std::string kernel_name,
                                         cl::Kernel *kernel) {
    cl_int err = 0;
    *kernel = std::move(cl::Kernel(program, kernel_name.c_str(), &err));
    ASSERT_EQ(0, err);
}

void BareMetalOpenCLTest::TearDown() {}


/* static */
void BareMetalOpenCLTest::ReadFile(const std::string &file_name,
		std::string *content) {
    FILE *file = fopen(file_name.c_str(), "rb");
    ASSERT_NE(file, nullptr);
    char buf[1024];
    content->clear();
    while (true) {
    	const int bytes = fread(buf, 1, sizeof(buf), file);
    	content->append(buf, bytes);
    	if (bytes != sizeof(buf)) {
    		break;
    	}
    }
    fclose(file);
}

} // namespace crater::opencl
