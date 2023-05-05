#include "parser.h"
#include "amdgpu_program.h"
#include "opencl/hsa/assert.h"

#include <absl/types/span.h>
#include <cstring>
#include <memory>

namespace ocl::hip {

ELFParserBase::ELFParserBase() : header_(nullptr) {}

absl::Status ELFParserBase::ParseHeaders() {
    if (blob_.size() <
        header_->e_shoff + header_->e_shnum * sizeof(Elf64SectionHeader)) {
        return absl::InvalidArgumentError("Malformed ELF header");
    }

    auto segments = reinterpret_cast<const Elf64ProgramHeader *>(
        blob_.data() + header_->e_phoff);
    std::copy(segments, segments + header_->e_phnum,
              std::back_inserter(segments_));

    result_->load_segments_.clear();
    for (const auto &s : segments_) {
        if (s.p_type != ELFSegmentType::PT_LOAD) {
            continue;
        }
        AMDGPUProgram::Segment r = {.file_offset = s.p_offset,
                                    .file_size = s.p_filesz,
                                    .vma_start = s.p_vaddr,
                                    .vma_length = s.p_memsz};
        result_->load_segments_.push_back(r);
    }

    auto sections = reinterpret_cast<const Elf64SectionHeader *>(
        blob_.data() + header_->e_shoff);
    std::copy(sections, sections + header_->e_shnum,
              std::back_inserter(section_headers_));

    const auto &desc = section_headers_.at(header_->e_shstrndx);
    if (blob_.size() < desc.sh_offset + desc.sh_size) {
        return absl::InvalidArgumentError("Malformed section");
    }

    section_string_table_ =
        std::string_view(blob_.data() + desc.sh_offset, desc.sh_size);
    return absl::OkStatus();
}

absl::Status ELFParserBase::VMAToFileOffset(uint64_t virtual_address,
                                            size_t *file_offset) {
    for (const auto &s : segments_) {
        if (s.p_vaddr > virtual_address ||
            virtual_address > s.p_vaddr + s.p_filesz) {
            continue;
        }
        *file_offset = virtual_address - s.p_vaddr + s.p_offset;
        return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Virtual address can not correspond to an offset of file image");
}

absl::Status ELFParserBase::ParseSections() {
    enum { kMaxSectionNameLength = 128 };
    static const char *kSectionName[] = {
        ".text", ".rodata", ".strtab", ".note", ".symtab",
    };
    for (const auto &s : section_headers_) {
        if (blob_.size() < s.sh_offset + s.sh_size) {
            return absl::InvalidArgumentError("Section out of bounds");
        } else if (s.sh_name >= section_string_table_.size()) {
            return absl::InvalidArgumentError("Section name out of bounds");
        }

        std::string_view name = GetCStr(
            kMaxSectionNameLength, section_string_table_.substr(s.sh_name));
        for (size_t i = 0; i < Section::kTotal; i++) {
            if (name == kSectionName[i]) {
                sections_[i] =
                    std::string_view(blob_.data() + s.sh_offset, s.sh_size);
                break;
            }
        }
    }
    return absl::OkStatus();
}

std::string_view ELFParserBase::GetBlobInSection(unsigned section_idx,
                                                 size_t offset,
                                                 size_t expected_size) {
    if (section_idx > section_headers_.size()) {
        return std::string_view();
    }
    const auto &s = section_headers_[section_idx];
    if (blob_.size() <= s.sh_offset + offset + expected_size) {
        return std::string_view();
    }
    return std::string_view(blob_.data() + s.sh_offset + offset, expected_size);
}

std::string_view ELFParserBase::GetBlob(size_t file_offset,
                                        size_t expected_size) {
    if (blob_.size() <= file_offset + expected_size) {
        return std::string_view();
    }
    return std::string_view(blob_.data() + file_offset, expected_size);
}

std::string_view ELFParserBase::GetCStr(size_t max_length,
                                        std::string_view base) {
    size_t len = 0;
    size_t end = std::min<size_t>(max_length, base.size());
    while (len < end && base.data()[len]) {
        len++;
    }
    return base.substr(0, len);
}

//
// Guessing the length of the ELF. It is required to implement
// hipModuleLoadData(). Try to guess the size as LLVM / ROCm seems to always
// emit the section header at the end of the file.
size_t ELFParserBase::GuessELFBinarySize(const char *data) {
    auto h = reinterpret_cast<const Elf64Header *>(data);
    auto length = h->e_shoff + (h->e_shnum * h->e_shentsize);
    return length;
}

} // namespace ocl::hip
