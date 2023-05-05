#include "platform.h"
#include "enclave/guest_platform.h"
#include "g6/g6_platform.h"
#include "kfd/kfd_platform.h"

#include <hsa/hsakmttypes.h>

namespace ocl::hsa {

Platform::Variant Platform::variant_;
void Platform::ChooseVariant(Variant variant) { variant_ = variant; }

Platform &Platform::Instance() {
    switch (variant_) {
    case kPlatformG6:
        return G6Platform::Instance();
    case kPlatformEnclaveGuest:
        return enclave::EnclaveGuestPlatform::Instance();
    default:
        return KFDPlatform::Instance();
    }
}

std::unique_ptr<Event> Platform::NewSignalEvent() {
    return NewEvent(HSA_EVENTTYPE_SIGNAL, 0);
}

} // namespace ocl::hsa