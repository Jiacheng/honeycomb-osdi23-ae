#pragma once

#include <CL/cl_wrap.hpp>
#include <gtest/gtest.h>

namespace crater::opencl {

class SimpleOpenCLTestContext {
  public:
    int Initialize(const std::string &blob_filename, const std::string &kernel);
    cl::CommandQueue &GetQueue() { return queue_; }
    cl::Kernel &GetKernel() { return kernel_; }
    cl::Context &GetContext() { return ctx_; }
    cl::Device &GetDevice() { return device_; }
    cl::Program &GetProgram() { return prog_; }

  private:
    cl::Context ctx_;
    cl::CommandQueue queue_;
    cl::Program prog_;
    cl::Kernel kernel_;
    cl::Device device_;
};

std::vector<cl::Device> EnumerateAMDGPUs();

class BareMetalOpenCLTest : public ::testing::Test {
  protected:
    virtual void SetUp() override;
    virtual void TearDown() override;
    cl::CommandQueue &GetQueue() { return queue_; }
    cl::Context &GetContext() { return ctx_; }
    cl::Device &GetDevice() { return device_; }

    void MustLoadBinaryFromFile(const std::string &filename, cl::Program *prog);
    void MustLoadKernel(const cl::Program &program,
                        const std::string kernel_name, cl::Kernel *kernel);

    static void ReadFile(const std::string &file_name, std::string *content);

    cl::Device device_;
    cl::Context ctx_;
    cl::CommandQueue queue_;
};

} // namespace crater::opencl
