#pragma once

#include "idl.h"
#include <absl/types/span.h>
#include <atomic>

namespace ocl::hsa::enclave {

class TransmitBuffer {
  public:
    enum { kMaxRPCSize = 65536 };
    explicit TransmitBuffer(absl::Span<char> buf, std::atomic_size_t *rptr,
                            std::atomic_size_t *wptr);
    char *ReadPacketAt(size_t rptr, idl::RPCType *ty, size_t *payload_size,
                       absl::Span<char> tmp);
    void Push(unsigned type_tag, absl::Span<const char> resp);
    absl::Span<char> GetBuffer() const { return buf_; }
    std::atomic_size_t *GetRptr() const { return rptr_; }
    std::atomic_size_t *GetWptr() const { return wptr_; }

  private:
    absl::Span<char> buf_;
    std::atomic_size_t *rptr_;
    std::atomic_size_t *wptr_;
};

} // namespace ocl::hsa::enclave