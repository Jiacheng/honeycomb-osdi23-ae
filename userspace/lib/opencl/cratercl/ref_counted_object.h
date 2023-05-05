#pragma once

namespace crater::opencl {

//
// Temporary hacks to implement reference count objects.
// A more diciplined approach might be to implement something like
// std::enable_shared_from_this / std::shared_ptr while exposing the ref counts.
template <class T> class RefCountedObject {
  public:
    void Retain() { ref_count_++; }
    void Release() {
        if (--ref_count_ == 0) {
            delete static_cast<T *>(this);
        }
    }

  protected:
    RefCountedObject(const RefCountedObject &) = delete;
    RefCountedObject &operator=(const RefCountedObject &) = delete;
    RefCountedObject() : ref_count_(1) {}
    unsigned ref_count_;
};
} // namespace crater::opencl