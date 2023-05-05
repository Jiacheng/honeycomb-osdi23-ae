#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void keccak_256(unsigned char *ret, const unsigned char *data, size_t size);
void keccak_512(unsigned char *ret, const unsigned char *data, size_t size);
void SHA3_256(unsigned char *ret, const unsigned char *data, size_t size);
void SHA3_512(unsigned char *ret, const unsigned char *data, size_t size);

#ifdef __cplusplus
}
#endif
