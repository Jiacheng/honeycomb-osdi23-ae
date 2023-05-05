#include "device_context.h"
#include "opencl/hsa/platform.h"
#include "opencl/hsa/runtime_options.h"
#include "utils/monad_runner.h"
#include <absl/status/status.h>
#include <memory>

#include "opencl/hsa/enclave/guest_platform.h"

namespace ocl::hip {

DeviceContext::DeviceContext() : initialized_(false) {
    auto stat = Initialize();
    initialized_ = stat.ok();
}

absl::Status DeviceContext::Initialize() {
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    auto opt = GetRuntimeOptions();
    if (!opt) {
        return absl::InvalidArgumentError("can not get runtime options");
    }
    if (opt->IsEnclave()) {
        hsa::Platform::ChooseVariant(hsa::Platform::kPlatformEnclaveGuest);
    }
    auto &plat = hsa::Platform::Instance();
    std::unique_ptr<HipMemoryManagerBase> mm;
    std::unique_ptr<HipComputeContextBase> compute;
    runner
        .Run([&]() {
            if (opt->IsEnclave()) {
                auto &enclave_plat =
                    static_cast<hsa::enclave::EnclaveGuestPlatform &>(plat);
                hsa::enclave::EnclaveGuestPlatform::Options options;
                options.shm_fn = opt->GetEnclaveShm();
                enclave_plat.SetOptions(options);
            }
            return plat.Initialize();
        })
        .Run([&]() {
            if (plat.GetDevices().empty()) {
                return absl::ResourceExhaustedError("Cannot get device");
            }
            impl_ = plat.GetDevices()[0];
            signal_pool_ =
                std::make_unique<hsa::SignalPool>(impl_->GetMemoryManager());
            barrier_ = signal_pool_->GetSignal();
            return absl::OkStatus();
        })
        .Run([&]() {
            mm = HipMemoryManagerBase::NewHsaMemoryManager(this, barrier_);
            return mm->Initialize();
        })
        .Run([&]() {
            compute =
                HipComputeContextBase::NewHsaComputeContext(this, barrier_);
            return compute->Initialize();
        })
        .Run([&]() {
            mm_ = std::move(mm);
            compute_ = std::move(compute);
            return absl::OkStatus();
        })
        .Run([&]() {
            if (opt->IsSecureMemcpy()) {
                usm_ = std::make_unique<HipSecureMemcpy>(HipSecureMemcpy());
                return usm_->Initialize(this);
            } else {
                return absl::OkStatus();
            }
        });
    return runner.code();
}

DeviceContext::~DeviceContext() {
    if (usm_) {
        usm_->Destroy();
        usm_.reset();
    }
    if (barrier_) {
        signal_pool_->PutSignal(barrier_);
        barrier_ = nullptr;
    }
    if (signal_pool_) {
        signal_pool_->Destroy();
    }
    if (compute_) {
        auto _ = compute_->Destroy();
        (void)_;
        compute_.reset();
    }
    if (mm_) {
        auto _ = mm_->Destroy();
        (void)_;
        mm_.reset();
    }
}

DeviceContext *GetCurrentDeviceContext() {
    static DeviceContext inst;
    return inst.IsInitialized() ? &inst : nullptr;
}

} // namespace ocl::hip
