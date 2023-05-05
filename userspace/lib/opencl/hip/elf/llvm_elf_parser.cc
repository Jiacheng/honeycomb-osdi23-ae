// #include <absl/types/span.h>

#include "amdgpu_program.h"
#include "code_object_v3_metadata_parser.h"
#include "parser.h"
#include "utils/monad_runner.h"
#include "utils/align.h"

namespace ocl::hip {

//
// The ELF parser for the binary generated from ROCm / LLVM.
// No relocation is required. The runtime just need to load
// the full image to the GPU device memory.
class LLVMELFParser : public ELFParserBase {
  public:
    explicit LLVMELFParser(AMDGPUProgram *result);
    static bool IsLLVMFormat(const Elf64Header *header);
    static std::unique_ptr<AMDGPUProgram> ParseBinary(std::string_view blob,
                                                      absl::Status *ret);

  private:
    absl::Status Parse(std::string_view blob);
    absl::Status CheckELFMagic();
    absl::Status ParseKernelDescriptorVMA();
    absl::Status ParseKernelMetadata();

    // ELF symbols of kernel descriptor => VMA
    std::map<std::string, uint64_t> kd_vmas_;
};

LLVMELFParser::LLVMELFParser(AMDGPUProgram *result) : ELFParserBase(result) {}

bool LLVMELFParser::IsLLVMFormat(const Elf64Header *header) {
    return header->e_machine == ELFMachine::EM_AMDGPU &&
           (header->e_ident[EI_ABIVERSION] == 1 ||
            header->e_ident[EI_ABIVERSION] == 2) &&
           header->e_flags;
}

absl::Status LLVMELFParser::CheckELFMagic() {
    header_ = reinterpret_cast<const Elf64Header *>(blob_.data());
    if (!IsLLVMFormat(header_)) {
        return absl::InvalidArgumentError("Invalid binary");
    }
    return absl::OkStatus();
}

absl::Status LLVMELFParser::ParseKernelDescriptorVMA() {
    enum { kMaxSymbolLength = 4096 };
    absl::Span<const Elf64Symbol> symbols(
        reinterpret_cast<Elf64Symbol *>(
            const_cast<char *>(sections_[kSymbolTable].data())),
        sections_[kSymbolTable].size() / sizeof(Elf64Symbol));

    for (const auto &s : symbols) {
        auto symbol_name =
            GetCStr(kMaxSymbolLength, sections_[kStrtab].substr(s.st_name));
        kd_vmas_[std::string(symbol_name)] = s.st_value;
    }
    return absl::OkStatus();
}

absl::Status LLVMELFParser::ParseKernelMetadata() {
    const auto s = sections_[Section::kNote];
    const auto size = s.size();

    size_t next = 0;
    for (size_t offset = 0; offset + sizeof(Elf64Note) < size; offset = next) {
        const auto note =
            reinterpret_cast<const Elf64Note *>(s.data() + offset);
        auto aligned_name_size = gpumpc::AlignUp(note->n_namesz, 4);
        next = offset + sizeof(Elf64Note) +
               gpumpc::AlignUp(aligned_name_size + note->n_descsz, 4);

        if (offset + note->n_descsz + aligned_name_size >= size) {
            return absl::InvalidArgumentError("Notes out of bounds");
        } else if (note->n_namesz != 7) {
            // Only support Code Object V3
            continue;
        }

        auto name = s.substr(offset + sizeof(Elf64Note), 6);
        if (name != "AMDGPU") {
            continue;
        }

        auto md = s.substr(offset + +sizeof(Elf64Note) + aligned_name_size,
                           note->n_descsz);
        CodeObjectV3MetadataParser parser(kd_vmas_);
        auto stat = parser.Parse(md, &result_->kernels_);
        if (!stat.ok()) {
            return stat;
        }
    }

    return absl::OkStatus();
}

absl::Status LLVMELFParser::Parse(std::string_view blob) {
    blob_ = blob;
    gpumpc::MonadRunner<absl::Status> runner(absl::OkStatus());
    runner.Run([this]() { return CheckELFMagic(); })
        .Run([this]() { return ParseHeaders(); })
        .Run([this]() { return ParseSections(); })
        .Run([this]() { return ParseKernelDescriptorVMA(); })
        .Run([this]() { return ParseKernelMetadata(); });
    return runner.code();
}

std::unique_ptr<AMDGPUProgram> LLVMELFParser::ParseBinary(std::string_view blob,
                                                          absl::Status *ret) {
    auto result = std::make_unique<AMDGPUProgram>();
    LLVMELFParser parser(result.get());
    *ret = parser.Parse(blob);
    if (!ret->ok()) {
        return std::unique_ptr<AMDGPUProgram>();
    }
    return result;
}

std::unique_ptr<AMDGPUProgram> ParseAMDGPUProgram(std::string_view blob,
                                                  absl::Status *ret) {
    if (blob.size() < sizeof(Elf64Header)) {
        *ret = absl::InvalidArgumentError("Malformed binary");
        return std::unique_ptr<AMDGPUProgram>();
    }
    const auto header = reinterpret_cast<const Elf64Header *>(blob.data());
    if (LLVMELFParser::IsLLVMFormat(header)) {
        return LLVMELFParser::ParseBinary(blob, ret);
    }
    *ret = absl::InvalidArgumentError("Unrecognized binary format");
    return std::unique_ptr<AMDGPUProgram>();
}

} // namespace ocl::hip
