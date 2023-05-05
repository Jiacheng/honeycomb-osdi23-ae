#include "cl_kernel.h"
#include "api.h"
#include "cl_memobj.h"
#include "cl_program.h"

namespace crater::opencl {

cl_kernel Kernel::clCreateKernel(cl_program program, const char *kernel_name,
                                 cl_int *errcode_ret) {
    auto prog = static_cast<Program *>(program);
    auto hip = prog->GetContext()->GetCtx();
    auto hip_module = prog->GetImpl();

    hipFunction_t kernel;
    auto error = hip->GetComputeContext()->hipModuleGetFunction(
        &kernel, hip_module, kernel_name);

    if (error != hipSuccess) {
        Nullable(errcode_ret).Assign(CL_INVALID_KERNEL_NAME);
        return nullptr;
    }

    Nullable(errcode_ret).Assign(CL_SUCCESS);
    prog->Retain();
    return new Kernel(prog, kernel);
}

cl_int Kernel::clSetKernelArg(cl_kernel kernel, cl_uint arg_index,
                              size_t arg_size, const void *arg_value) {
    auto self = static_cast<Kernel *>(kernel);
    return self->SetKernelArg(arg_index, arg_size, arg_value);
}

cl_int Kernel::clRetainKernel(cl_kernel kernel) {
    auto self = static_cast<Kernel *>(kernel);
    self->Retain();
    return CL_SUCCESS;
}

cl_int Kernel::clReleaseKernel(cl_kernel kernel) {
    auto self = static_cast<Kernel *>(kernel);
    self->Release();
    return CL_SUCCESS;
}

Kernel::Kernel(Program *program, hipFunction_t kernel)
    : parent_(program), kernel_(kernel), args_buffer_(nullptr),
      args_buffer_size_(0) {
    ComputeArgumentPosition();
}

Kernel::~Kernel() {
    if (args_buffer_) {
        delete[] args_buffer_;
    }
    parent_->Release();
}

void Kernel::ComputeArgumentPosition() {
    auto hip_kernel = reinterpret_cast<const ocl::hip::Kernel *>(kernel_);
    const auto &info = hip_kernel->GetKernelInfo();
    args_.resize(info->args.size());
    // get total arg buffer size
    if (info->kernarg_size) {
        args_buffer_size_ = info->kernarg_size;
    } else {
        args_buffer_size_ = 0;
        for (size_t i = 0; i < info->args.size(); ++i) {
            args_buffer_size_ = std::max(args_buffer_size_,
                                         info->args[i].GetOffset()
                                         + info->args[i].GetSize());
        }
    }
    
    args_buffer_ = new char[args_buffer_size_];
    memset(args_buffer_, 0, args_buffer_size_);
    for (size_t i = 0; i < info->args.size(); ++i) {
        HSA_ASSERT(info->args[i].GetOffset() + info->args[i].GetSize() <= args_buffer_size_);
        args_[i] = absl::Span<char>(args_buffer_ + info->args[i].GetOffset(),
                                    info->args[i].GetSize());
    }
}

cl_int Kernel::SetKernelArg(cl_uint arg_index, size_t arg_size,
                            const void *arg_value) {
    std::lock_guard<std::recursive_mutex> lk(mutex_);
    // NOTE: always assume LLVM Calling Convention
    size_t real_idx = arg_index;
    // NOTE: Implicit argument may not always be in
    // if (real_idx >=
    //    args_.size() - ocl::hip::KernelArgument::kImplicitArgumentNumLLVM) {
    //    return CL_INVALID_ARG_INDEX;
    //}

    if (arg_value && args_[real_idx].size() != arg_size) {
        return CL_INVALID_ARG_SIZE;
    }

    auto hip_kernel = reinterpret_cast<const ocl::hip::Kernel *>(kernel_);
    const auto &info = hip_kernel->GetKernelInfo();
    if (!arg_value) {
        memset(args_[real_idx].data(), 0,
               std::min(arg_size, args_[real_idx].size()));
    } else if (info->args[real_idx].GetType() ==
               ocl::hip::KernelArgument::kArgPointer) {
        auto arg =
            reinterpret_cast<ocl::hsa::gpu_addr_t *>(args_[real_idx].data());
        auto buffer = *reinterpret_cast<const Memory *const *>(arg_value);
        if (buffer) {
            auto gpu_addr = buffer->GetGPUAddress();
            *arg = gpu_addr;
        } else {
            *arg = 0;
        }
    } else {
        memcpy(args_[real_idx].data(), arg_value, arg_size);
    }

    return CL_SUCCESS;
}

} // namespace crater::opencl
