#pragma once

namespace ocl::hip {

enum {
    kAESBlockSize = 16,
    kAESBlockSizeInWord = 4,
    kAESBlockSizeLog2 = 4,
    kAESIVSize = 16,
    kAESIVSizeInWord = 4,
    kAES256KeySize = 32,
    kAES256KeySizeInWord = 8,
};

}