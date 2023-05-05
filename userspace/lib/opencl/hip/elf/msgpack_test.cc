#include "msgpack.h"
#include <gtest/gtest.h>

TEST(TestMsgPack, TestUnpack) {
    using namespace ocl::hip;
    static const unsigned char kRepr[] = {0x82, 0xa7, 0x63, 0x6f, 0x6d, 0x70,
                                          0x61, 0x63, 0x74, 0xc3, 0xa6, 0x73,
                                          0x63, 0x68, 0x65, 0x6d, 0x61, 0x00};
    absl::Status stat;
    auto obj = msgpack::Object::Unpack(
        std::string_view(reinterpret_cast<const char *>(kRepr), sizeof(kRepr)),
        &stat);
    ASSERT_TRUE(stat.ok());
    ASSERT_EQ(msgpack::Object::Type::kMap, obj->GetType());
}
