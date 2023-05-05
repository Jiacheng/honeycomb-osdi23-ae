#pragma once

#include <absl/status/status.h>

#include <memory>
#include <string_view>
#include <vector>

namespace ocl::hip::msgpack {

class Object {
  public:
    friend class Unpacker;
    enum Type { kUnknown, kNil, kInteger, kString, kMap, kArray };
    virtual Type GetType() const = 0;
    virtual ~Object() = default;
    static std::unique_ptr<Object> Unpack(std::string_view data,
                                          absl::Status *err);
    template <class T> static inline T *dyn_cast(Object *o) {
        return (o->GetType() == T::kType) ? static_cast<T *>(o) : nullptr;
    }

  protected:
    Object(const Object &) = delete;
    Object &operator=(const Object &) = delete;
    Object() = default;
};

class Nil : public Object {
  public:
    friend class Unpacker;
    static const Type kType = kNil;
    virtual Type GetType() const override { return kType; }

  private:
    Nil() = default;
};

class String : public Object {
  public:
    friend class Unpacker;
    static const Type kType = kString;
    virtual Type GetType() const override { return kType; }
    std::string_view GetValue() const { return data_; }

  private:
    explicit String(const std::string_view &s);
    const std::string_view data_;
};

class Integer : public Object {
  public:
    friend class Unpacker;
    static const Type kType = kInteger;
    virtual Type GetType() const override { return kType; }
    bool IsUnsigned() const { return is_unsigned_; }
    uint64_t GetValue() const { return value_; }

  private:
    explicit Integer(bool is_unsigned, uint64_t value);
    const bool is_unsigned_;
    const uint64_t value_;
};

class Map : public Object {
  public:
    friend class Unpacker;
    static const Type kType = kMap;
    virtual Type GetType() const override { return kType; }
    ~Map();
    const std::vector<std::pair<Object *, Object *>> &GetValue() const {
        return entries_;
    }

  private:
    explicit Map(std::vector<std::pair<Object *, Object *>> &&entries);
    std::vector<std::pair<Object *, Object *>> entries_;
};

class Array : public Object {
  public:
    friend class Unpacker;
    static const Type kType = kArray;
    virtual Type GetType() const override { return kType; }
    ~Array();
    const std::vector<Object *> &GetValue() const { return entries_; }

  private:
    explicit Array(std::vector<Object *> &&entries);
    std::vector<Object *> entries_;
};

} // namespace ocl::hip::msgpack
