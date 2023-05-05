#pragma once

#include "elf.h"

#include <absl/status/status.h>

#include <map>
#include <memory>
#include <string_view>
#include <vector>

namespace ocl::hip {

class AMDGPUProgram;

class ELFParserBase {
  public:
    static size_t GuessELFBinarySize(const char *data);

  protected:
    struct KernelRelocationInfo {
        size_t size;
        std::string name;
    };

    enum Section {
        kText,
        kROData,
        kStrtab,
        kNote,
        kSymbolTable,
        kTotal,
    };

    ELFParserBase();
    absl::Status ParseHeaders();
    absl::Status VMAToFileOffset(uint64_t virtual_address, size_t *file_offset);
    absl::Status ParseSections();
    std::string_view GetBlobInSection(unsigned section_idx, size_t offset,
                                      size_t expected_size);
    std::string_view GetBlob(size_t file_offset, size_t expected_size);
    std::string_view GetCStr(size_t max_length, std::string_view base);
    absl::Status ParseKernelCode(const std::string &kernel_name,
                                 unsigned section_idx, uint64_t virtual_address,
                                 size_t code_size);
    absl::Status ParseROData();
    explicit ELFParserBase(AMDGPUProgram *result) : result_(result) {}

    std::string_view blob_;
    const Elf64Header *header_;
    std::string_view section_string_table_;
    std::array<std::string_view, Section::kTotal> sections_;

    std::vector<Elf64ProgramHeader> segments_;
    std::vector<Elf64SectionHeader> section_headers_;
    std::map<size_t, KernelRelocationInfo> kernel_locations_;
    AMDGPUProgram *result_;
};

} // namespace ocl::hip
