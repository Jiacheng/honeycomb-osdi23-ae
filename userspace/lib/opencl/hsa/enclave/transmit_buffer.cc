#include "transmit_buffer.h"
#include "opencl/hsa/assert.h"
#include "opencl/hsa/enclave/idl.h"
#include <cstring>
#include <utility>

namespace ocl::hsa::enclave {

using idl::RPCType;

TransmitBuffer::TransmitBuffer(absl::Span<char> buf, std::atomic_size_t *rptr,
                               std::atomic_size_t *wptr)
    : buf_(buf), rptr_(rptr), wptr_(wptr) {}

char *TransmitBuffer::ReadPacketAt(size_t rptr, idl::RPCType *ty,
                                   size_t *payload_size, absl::Span<char> tmp) {
    if (tmp.size() < kMaxRPCSize) {
        return nullptr;
    }
    auto base = buf_.data();
    auto size = buf_.size();

    unsigned type_tag = *reinterpret_cast<unsigned *>(base + rptr);
    *ty = (idl::RPCType)(type_tag & 0xff);
    *payload_size = idl::GetPayloadSize(type_tag);
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

void TransmitBuffer::Push(unsigned type_tag, absl::Span<const char> payload) {
    auto rptr = rptr_->load();
    auto wptr = wptr_->load();
    auto base = buf_.data();
    auto size = buf_.size();

    while (wptr + payload.size() - rptr >= size) {
        sched_yield();
        rptr = rptr_->load();
    }

    if (wptr + sizeof(type_tag) + payload.size() <= size) {
        memcpy(base + wptr, &type_tag, sizeof(type_tag));
        memcpy(base + wptr + sizeof(type_tag), payload.data(), payload.size());
    } else {
        HSA_ASSERT(wptr + sizeof(type_tag) <= size);
        memcpy(base + wptr, &type_tag, sizeof(type_tag));
        size_t p = size - wptr - sizeof(type_tag);
        memcpy(base + wptr + sizeof(type_tag), payload.data(), p);
        memcpy(base, payload.data() + p, payload.size() - p);
    }

    wptr = (wptr + sizeof(type_tag) + payload.size()) & (size - 1);
    wptr_->store(wptr);
}

} // namespace ocl::hsa::enclave
