#include "msgpack.h"

#include <memory>
#include <sstream>
#include <vector>

namespace ocl::hip::msgpack {

String::String(const std::string_view &s) : data_(s) {}

Integer::Integer(bool is_unsigned, uint64_t value)
    : is_unsigned_(is_unsigned), value_(value) {}

Map::Map(std::vector<std::pair<Object *, Object *>> &&entries)
    : entries_(std::move(entries)) {}

Map::~Map() {
    for (const auto &e : entries_) {
        delete e.first;
        delete e.second;
    }
}

Array::Array(std::vector<Object *> &&entries) : entries_(std::move(entries)) {}
Array::~Array() {
    for (const auto &e : entries_) {
        delete e;
    }
}

class Unpacker {
  public:
    explicit Unpacker(const std::string_view data)
        : data_(reinterpret_cast<const unsigned char *>(data.data())),
          size_(data.size()), offset_(0) {}

    std::unique_ptr<Object> Parse(absl::Status *err);
    static Object::Type DetermineType(unsigned char c, absl::Status *err);
    std::unique_ptr<Object> ParseInteger(unsigned char c, absl::Status *err);
    std::unique_ptr<Object> ParseString(unsigned char c, absl::Status *err);
    std::unique_ptr<Object> ParseMap(unsigned char c, absl::Status *err);
    std::unique_ptr<Object> ParseArray(unsigned char c, absl::Status *err);

  private:
    bool eof() const { return offset_ >= size_; }
    uint64_t Available() const {
        return offset_ >= size_ ? 0 : size_ - offset_;
    }
    unsigned char Peek() const { return data_[offset_]; }
    const unsigned char *Current() const { return &data_[offset_]; }
    void Advance(unsigned c) { offset_ += c; }
    uint64_t ReadBE(size_t length);

    const unsigned char *data_;
    const size_t size_;
    size_t offset_;
};

std::unique_ptr<Object> Unpacker::Parse(absl::Status *err) {
    if (eof()) {
        *err = absl::InvalidArgumentError("Out of bounds");
        return std::unique_ptr<Object>();
    }
    auto c = Peek();
    auto r = DetermineType(Peek(), err);
    if (!err->ok()) {
        return std::unique_ptr<Object>();
    }
    Advance(1);

    switch (r) {
    case Object::Type::kNil: {
        *err = absl::OkStatus();
        return std::unique_ptr<Object>(new Nil());
    }
    case Object::Type::kInteger:
        return ParseInteger(c, err);
    case Object::Type::kString:
        return ParseString(c, err);
    case Object::Type::kMap:
        return ParseMap(c, err);
    case Object::Type::kArray:
        return ParseArray(c, err);
    default:
    case Object::Type::kUnknown:
        return std::unique_ptr<Object>(new Nil());
    }
}

uint64_t Unpacker::ReadBE(size_t length) {
    union {
        char repr[8];
        uint64_t r;
    };
    r = 0;
    auto c = Current();
    for (size_t i = 0; i < length; ++i) {
        repr[i] = c[length - 1 - i];
    }
    Advance(length);
    return r;
}

Object::Type Unpacker::DetermineType(unsigned char c, absl::Status *err) {
    *err = absl::OkStatus();
    if (c <= 0x7f || (0xc2 <= c && c <= 0xc3) || (0xcc <= c && c <= 0xd3) ||
        (0xe0 <= c && c <= 0xff)) {
        return Object::Type::kInteger;
    } else if ((0xa0 <= c && c <= 0xbf) || (0xd9 <= c && c <= 0xdb)) {
        return Object::Type::kString;
    } else if ((0x80 <= c && c <= 0x8f) || (0xde <= c && c <= 0xdf)) {
        return Object::Type::kMap;
    } else if ((0x90 <= c && c <= 0x9f) || (0xdc <= c && c <= 0xdd)) {
        return Object::Type::kArray;
    }
    *err = absl::InvalidArgumentError("Unknown type " + std::to_string(c));
    return Object::Type::kUnknown;
}

std::unique_ptr<Object> Unpacker::ParseInteger(unsigned char c, absl::Status *err) {
    *err = absl::OkStatus();
    if (c <= 0x7f || (0xe0 <= c && c <= 0xff)) {
        auto is_unsigned = c <= 0x7f;
        return std::unique_ptr<Object>(
            new Integer(is_unsigned, is_unsigned ? c : (int64_t)(char)c));
    } else if (0xc2 <= c && c <= 0xc3) {
        // Boolean
        return std::unique_ptr<Object>(new Integer(true, c == 0xc3));
    } else {
        // (0xcc <= c && c <= 0xd3)
        auto length = 1u << ((c - 0xcc) % 4);
        auto is_unsigned = c <= 0xcf;
        if (Available() < length) {
            *err = absl::InvalidArgumentError("Malformed input");
            return std::unique_ptr<Object>();
        }
        auto v = ReadBE(length);
        if (is_unsigned) {
            return std::unique_ptr<Object>(new Integer(is_unsigned, v));
        }
        int64_t signed_v;
        if (length == 1) {
            signed_v = *reinterpret_cast<const char *>(&v);
        } else if (length == 2) {
            signed_v = *reinterpret_cast<const short *>(&v);
        } else if (length == 4) {
            signed_v = *reinterpret_cast<const int *>(&v);
        } else {
            signed_v = *reinterpret_cast<int64_t *>(&v);
        }
        return std::unique_ptr<Object>(
            new Integer(is_unsigned, (uint64_t)signed_v));
    }
}

std::unique_ptr<Object> Unpacker::ParseString(unsigned char c, absl::Status *err) {
    size_t length = 0;
    if (0xa0 <= c && c <= 0xbf) {
        length = c & 0x1f;
    } else {
        auto l = 1u << ((c - 0xd9) % 4);
        if (Available() < l) {
            *err = absl::InvalidArgumentError("Malformed input");
            return std::unique_ptr<Object>();
        }
        length = ReadBE(l);
    }
    if (Available() < length) {
        *err = absl::InvalidArgumentError("Malformed input");
        return std::unique_ptr<Object>();
    }
    auto ret = std::unique_ptr<Object>(new String(
        std::string_view(reinterpret_cast<const char *>(Current()), length)));
    Advance(length);
    return ret;
}

std::unique_ptr<Object> Unpacker::ParseMap(unsigned char c, absl::Status *err) {
    size_t num_entries = 0;
    if (0x80 <= c && c <= 0x8f) {
        num_entries = c - 0x80;
    } else {
        auto l = 2u << (c - 0xde);
        if (Available() < l) {
            *err = absl::InvalidArgumentError("Malformed input");
            return std::unique_ptr<Object>();
        }
        num_entries = ReadBE(l);
    }

    std::vector<std::pair<std::unique_ptr<Object>, std::unique_ptr<Object>>>
        entries;
    for (size_t i = 0; i < num_entries; ++i) {
        auto key = Parse(err);
        if (!err) {
            return std::unique_ptr<Object>();
        }
        auto value = Parse(err);
        if (!err) {
            return std::unique_ptr<Object>();
        }
        entries.push_back(
            std::make_pair<std::unique_ptr<Object>, std::unique_ptr<Object>>(
                std::move(key), std::move(value)));
    }
    *err = absl::OkStatus();
    std::vector<std::pair<Object *, Object *>> released(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        released[i] = std::make_pair(entries[i].first.release(),
                                     entries[i].second.release());
    }
    return std::unique_ptr<Object>(new Map(std::move(released)));
}

std::unique_ptr<Object> Unpacker::ParseArray(unsigned char c, absl::Status *err) {
    size_t num_entries = 0;
    if (0x90 <= c && c <= 0x9f) {
        num_entries = c - 0x90;
    } else {
        auto l = 2u << (c - 0xdc);
        if (Available() < l) {
            *err = absl::InvalidArgumentError("Malformed input");
            return std::unique_ptr<Object>();
        }
        num_entries = ReadBE(l);
    }

    std::vector<std::unique_ptr<Object>> entries;
    for (size_t i = 0; i < num_entries; ++i) {
        auto v = Parse(err);
        entries.emplace_back(std::move(v));
    }
    *err = absl::OkStatus();
    std::vector<Object *> released(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        released[i] = entries[i].release();
    }
    return std::unique_ptr<Object>(new Array(std::move(released)));
}

std::unique_ptr<Object> Object::Unpack(std::string_view data, absl::Status *err) {
    Unpacker unpacker(data);
    return unpacker.Parse(err);
}

} // namespace ocl::hip::msgpack