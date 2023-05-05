#include "transmit_buffer.h"
#include "opencl/hsa/assert.h"
#include <cstring>
#include <utility>

namespace ocl::hsa::enclave {

using idl::RPCType;

TransmitBuffer::TransmitBuffer(absl::Span<char> buf, std::atomic_size_t *rptr,
                               std::atomic_size_t *wptr)
    : buf_(buf), rptr_(rptr), wptr_(wptr) {}

const char *TransmitBuffer::ReadPacketAt(size_t rptr, idl::RPCType *ty,
                                         size_t *payload_size,
                                         absl::Span<char> tmp) {
    if (tmp.size() < kMaxRPCSize) {
        return nullptr;
    }
    auto base = buf_.data();
    auto size = buf_.size();

    *ty = *reinterpret_cast<const RPCType *>(base + rptr);
    *payload_size = idl::GetPayloadSize(*ty);
    auto req = rptr + sizeof(RPCType);
    if (rptr + sizeof(RPCType) + *payload_size > size) {
        HSA_ASSERT(rptr + sizeof(RPCType) <= size);
        size_t p = size - rptr - sizeof(RPCType);
        memcpy(tmp.data(), base + req, p);
        memcpy(tmp.data() + p, base, *payload_size - p);
        return tmp.data();
    } else {
        return base + req;
    }
}

void TransmitBuffer::Push(idl::RPCType type, absl::Span<const char> payload) {
    auto rptr = rptr_->load();
    auto wptr = wptr_->load();
    auto base = buf_.data();
    auto size = buf_.size();

    while (wptr + payload.size() - rptr >= size) {
        sched_yield();
        rptr = rptr_->load();
    }

    if (wptr + sizeof(type) + payload.size() <= size) {
        memcpy(base + wptr, &type, sizeof(type));
        memcpy(base + wptr + sizeof(type), payload.data(), payload.size());
    } else {
        HSA_ASSERT(wptr + sizeof(type) <= size);
        memcpy(base + wptr, &type, sizeof(type));
        size_t p = size - wptr - sizeof(type);
        memcpy(base + wptr + sizeof(type), payload.data(), p);
        memcpy(base, payload.data() + p, payload.size() - p);
    }

    wptr = (wptr + sizeof(type) + payload.size()) & (size - 1);
    wptr_->store(wptr);
}

} // namespace ocl::hsa::enclave
