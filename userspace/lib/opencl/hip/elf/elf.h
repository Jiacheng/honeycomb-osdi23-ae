#pragma once

#include <cstdint>

namespace ocl::hip {

enum {
    EI_CLASS = 4,      // File class.
    EI_DATA = 5,       // Data encoding.
    EI_VERSION = 6,    // File version.
    EI_OSABI = 7,      // OS/ABI identification.
    EI_ABIVERSION = 8, // ABI version.
    EI_NIDENT = 16,
};

// Segment types.
enum ELFSegmentType {
    PT_NULL = 0,
    PT_LOAD = 1,
    PT_DYNAMIC = 2,
    PT_INTERP = 3,
    PT_NOTE = 4,
    PT_SHLIB = 5,
    PT_PHDR = 6,
    PT_TLS = 7,
};

enum ELFSectionType {
    SHT_NULL = 0,
    SHT_PROGBITS,
    SHT_SYMTAB,
    SHT_STRTAB,
    SHT_RELA,
    SHT_HASA,
    SHT_DYNAMIC,
    SHT_NOTE,
    SHT_NOBITS,
    SHT_REL,
    SHT_DYNSYM,
};

enum ELFMachine {
    EM_AMDGPU = 224,
};

enum {
    NT_AMDGPU_METADATA = 32,
};

#pragma pack(push, 1)
struct Elf64Header {
    unsigned char e_ident[EI_NIDENT];
    unsigned short e_type;
    unsigned short e_machine;
    unsigned e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    unsigned e_flags;
    unsigned short e_ehsize;
    unsigned short e_phentsize;
    unsigned short e_phnum;
    unsigned short e_shentsize;
    unsigned short e_shnum;
    unsigned short e_shstrndx;
};

struct Elf64ProgramHeader {
    unsigned p_type;
    unsigned p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
};

struct Elf64SectionHeader {
    unsigned sh_name;
    unsigned sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    unsigned sh_link;
    unsigned sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};

struct Elf64Symbol {
    uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
};

// Relocation entry with explicit addend.
struct Elf64Rela {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t r_addend;
};

// Node header for ELF64.
struct Elf64Note {
    uint32_t n_namesz;
    uint32_t n_descsz;
    uint32_t n_type;
};
#pragma pack(pop)

inline unsigned ELF64RSym(uint64_t r_info) { return r_info >> 32; }
inline unsigned ELF64RType(uint64_t r_info) { return r_info & 0xffffffffu; }

} // namespace ocl::hip
