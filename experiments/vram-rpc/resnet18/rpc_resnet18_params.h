#pragma once

namespace gpumpc {
namespace experiment {

enum {
    kResNet18RPCServerBlockSize = 512,
    kResNet18RPCClientBlockSize = 512,
    kResNet18InputImageSize = 602112,
    kResNet18ResultSize = 4000,
    kResNet18RPCResultOffset = 256 * 4096,
    kRPCSynIndicatorOffset =
        kResNet18RPCResultOffset / sizeof(unsigned long) - 2,
    kRPCTimeOffset = kResNet18RPCResultOffset / sizeof(unsigned long) - 1,
    kRPCResultOffset = kResNet18RPCResultOffset / sizeof(unsigned long)
};
}
} // namespace gpumpc