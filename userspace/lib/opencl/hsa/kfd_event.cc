#include "kfd_event.h"
#include "opencl/hsa/event.h"
#include "platform.h"
#include "utils.h"

#include <cstdint>
#include <hsa/hsakmttypes.h>
#include <hsa/kfd_ioctl.h>
#include <memory>

namespace ocl::hsa {
using absl::Status;

KFDEvent::KFDEvent(unsigned event_id, unsigned long hw_data2,
                   unsigned long hw_data3)
    : Event(event_id, hw_data2, hw_data3) {}

KFDEvent::~KFDEvent() {
    auto ret = Destroy();
    if (!ret.ok()) {
        spdlog::warn("Cannot destroy event");
    }
}

std::unique_ptr<Event> KFDEvent::New(int type, uint64_t event_page_handle) {
    struct kfd_ioctl_create_event_args args = {0};

    args.event_type = type;
    args.node_id = 0;
    args.auto_reset = true;
    args.event_page_offset = event_page_handle;

    int kfd_fd = Platform::Instance().GetKFDFD();
    if (kmtIoctl(kfd_fd, AMDKFD_IOC_CREATE_EVENT, &args)) {
        return nullptr;
    }

    auto event_buf = Platform::Instance().GetEventPageBase();
    uint64_t hw_data2 = 0;
    if (args.event_page_offset > 0 &&
        args.event_slot_index < KFD_SIGNAL_EVENT_LIMIT) {
        hw_data2 = reinterpret_cast<unsigned long>(
            event_buf + Platform::kEventSlotSizeBytes * args.event_slot_index);
    }

    auto e = std::unique_ptr<KFDEvent>(
        new KFDEvent(args.event_id, hw_data2, args.event_trigger_data));
    return e;
}

std::unique_ptr<Event> KFDEvent::NewSignalEvent() {
    return KFDEvent::New(HSA_EVENTTYPE_SIGNAL, 0);
}

Status KFDEvent::Wait(unsigned long ms) {
    struct kfd_event_data d = {
        {
            {{0}},
        },
        0,
    };
    d.event_id = event_id_;
    struct kfd_ioctl_wait_events_args args = {0};

    args.wait_for_all = true;
    args.timeout = ms;
    args.num_events = 1;
    args.events_ptr = reinterpret_cast<uint64_t>(&d);

    int kfd_fd = Platform::Instance().GetKFDFD();
    if (kmtIoctl(kfd_fd, AMDKFD_IOC_WAIT_EVENTS, &args) == -1) {
        return absl::InvalidArgumentError("Cannot wait on the event");
    } else if (args.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT) {
        // XXX: Should be a time out
        return absl::OkStatus();
    }

    return absl::OkStatus();
}

Status KFDEvent::Destroy() {
    if (!event_id_) {
        return absl::OkStatus();
    }

    struct kfd_ioctl_destroy_event_args args = {0};
    int kfd_fd = Platform::Instance().GetKFDFD();
    args.event_id = event_id_;
    event_id_ = 0;
    if (kmtIoctl(kfd_fd, AMDKFD_IOC_DESTROY_EVENT, &args)) {
        return absl::InvalidArgumentError("Cannot destroy event");
    }
    return absl::OkStatus();
}

Status KFDEvent::Notify() {
    struct kfd_ioctl_set_event_args args = {.event_id = event_id_};
    int kfd_fd = Platform::Instance().GetKFDFD();
    if (kmtIoctl(kfd_fd, AMDKFD_IOC_SET_EVENT, &args)) {
        return absl::InvalidArgumentError("Cannot notify event");
    }
    return absl::OkStatus();
}

} // namespace ocl::hsa