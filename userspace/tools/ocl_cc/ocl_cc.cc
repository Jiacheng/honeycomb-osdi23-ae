#include <CL/cl_wrap.hpp>

#include <absl/status/status.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define checkCLErrors(call)                                                    \
    do {                                                                       \
        call;                                                                  \
        if (err) {                                                             \
            std::cerr << "OpenCL Error '" << err << "' in func '"              \
                      << __FUNCTION__ << "' line " << __LINE__ << "\n";        \
        }                                                                      \
    } while (0)

static const char kAMDPlatformName[] = "AMD Accelerated Parallel Processing";

static void ParseFlags(int argc, char *argv[], std::string *flags,
                       std::string *input, std::string *output) {
    int i = 1;
    std::stringstream ss;
    while (i < argc) {
        std::string f(argv[i]);
        if (f[0] == '-') {
            if (f == "-c") {
            } else if (f == "-o" && i + 1 < argc) {
                *output = argv[i + 1];
                ++i;
            } else {
                ss << f << " ";
            }
        } else {
            *input = f;
        }
        ++i;
    }
    *flags = ss.str();
}

using absl::Status;

std::vector<char> ReadAll(const std::string &fn, Status *stat) {
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

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << argv[0] << " <compile flags> <input> -c -o <output>"
                  << std::endl;
        return 1;
    }

    std::string flags;
    std::string input;
    std::string output;
    ParseFlags(argc, argv, &flags, &input, &output);
    if (!input.size() || !output.size()) {
        std::cerr << "No input or output file is specified\n";
        return 1;
    }

    Status stat;
    std::vector<char> kernel_src = ReadAll(input, &stat);
    if (!stat.ok()) {
        std::cerr << "Failed to read the source file: " << stat.ToString()
                  << std::endl;
        return 1;
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform;

    for (auto p : platforms) {
        auto r = p.getInfo<CL_PLATFORM_NAME>();
        if (!strncmp(r.c_str(), kAMDPlatformName, sizeof(kAMDPlatformName))) {
            platform = p;
            break;
        }
    }

    if (!platform()) {
        std::cerr << "No platform available\n";
        return 1;
    }

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device;
    if (!devices.size()) {
        std::cerr << "No devices availble\n";
        return 1;
    }
    device = devices.back();

    cl::Program::Sources source(
        1, std::make_pair(kernel_src.data(), kernel_src.size()));

    cl_int err = 0;
    cl_context_properties ctx_prop[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)(platform()), 0};
    cl::Context ctx;
    checkCLErrors(ctx = cl::Context(device, ctx_prop, nullptr, nullptr, &err));

    cl::Program prog;
    checkCLErrors(prog = cl::Program(ctx, source, &err));

    err = prog.build({device}, flags.c_str());
    if (err) {
        std::cerr << "Failed to compile the program:\n"
                  << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                  << std::endl;
        return -1;
    }

    std::cerr << "Compiled on " << platform.getInfo<CL_PLATFORM_NAME>()
              << "::" << device.getInfo<CL_DEVICE_NAME>() << "\n";

    auto bin_size = prog.getInfo<CL_PROGRAM_BINARY_SIZES>();
    auto bin = prog.getInfo<CL_PROGRAM_BINARIES>();
    std::cout << bin_size[0] << std::endl;
    std::ofstream os(output);
    os.write(bin[0], bin_size[0]);
    os.close();
    return 0;
}
