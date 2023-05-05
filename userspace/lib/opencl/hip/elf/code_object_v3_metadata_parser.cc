#include "code_object_v3_metadata_parser.h"
#include "msgpack.h"
#include "string.h"

#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <set>

namespace ocl::hip {

using namespace msgpack;
CodeObjectV3MetadataParser::CodeObjectV3MetadataParser(
    const std::map<std::string, uint64_t> &kd_vmas)
    : kd_vmas_(kd_vmas) {}

absl::Status CodeObjectV3MetadataParser::Parse(
    std::string_view data,
    std::map<std::string, AMDGPUProgram::KernelInfo> *out) {
    absl::Status stat;
    auto obj = Object::Unpack(data, &stat);
    if (!stat.ok()) {
        return stat;
    }
    auto m = Object::dyn_cast<Map>(obj.get());
    if (!m) {
        return absl::InvalidArgumentError("Malformed metadata");
    }

    for (const auto &e : m->GetValue()) {
        auto key = Object::dyn_cast<String>(e.first);
        if (!key || key->GetValue() != "amdhsa.kernels") {
            continue;
        }
        auto kernels = Object::dyn_cast<Array>(e.second);
        if (!kernels) {
            return absl::InvalidArgumentError("Malformed metadata");
        }
        for (const auto &k : kernels->GetValue()) {
            auto m = Object::dyn_cast<Map>(k);
            if (!m) {
                return absl::InvalidArgumentError("Malformed metadata");
            }
            AMDGPUProgram::KernelInfo ki;
            std::string name;
            stat = ParseKernelInfo(m, &name, &ki);
            if (!stat.ok()) {
                return stat;
            }
            out->insert({name, ki});
        }
    }
    return absl::OkStatus();
}

absl::Status CodeObjectV3MetadataParser::ParseKernelInfo(
    const msgpack::Map *m, std::string *name, AMDGPUProgram::KernelInfo *ki) {
    bool has_name = false;

    for (const auto &e : m->GetValue()) {
        auto key = Object::dyn_cast<String>(e.first);
        if (!key) {
            continue;
        } else if (key->GetValue() == ".name") {
            auto value = Object::dyn_cast<String>(e.second);
            if (!value) {
                return absl::InvalidArgumentError(
                    "Expect a string in a kernel name");
            }
            has_name = true;
            *name = std::string(value->GetValue());
        } else if (key->GetValue() == ".symbol") {
            auto value = Object::dyn_cast<String>(e.second);
            if (!value) {
                return absl::InvalidArgumentError("Expect a string in .symbol");
            }

            auto it = kd_vmas_.find(std::string(value->GetValue()));
            if (it == kd_vmas_.end()) {
                return absl::InvalidArgumentError(
                    "Failed to find the descriptor");
            }
            ki->desc_vma_offset = it->second;
        } else if (key->GetValue() == ".args") {
            auto value = Object::dyn_cast<Array>(e.second);
            if (!value) {
                return absl::InvalidArgumentError(
                    "Expect an array as arguments");
            }
            std::vector<KernelArgument> args;
            for (const auto &a : value->GetValue()) {
                if (auto v = Object::dyn_cast<Map>(a)) {
                    absl::Status stat;
                    auto arg = ParseArgument(v, &stat);
                    if (!stat.ok()) {
                        return stat;
                    }
                    args.push_back(arg);
                } else {
                    return absl::InvalidArgumentError(
                        "Expect a map as an argument");
                }
            }
            ki->args = std::move(args);
        } else if (key->GetValue() == ".group_segment_fixed_size") {
            auto value = Object::dyn_cast<Integer>(e.second);
            if (!value) {
                return absl::InvalidArgumentError("Expect an integer");
            }
            ki->lds_size = (unsigned)value->GetValue();
        } else if (key->GetValue() == ".kernarg_segment_align") {
            auto value = Object::dyn_cast<Integer>(e.second);
            if (!value) {
                return absl::InvalidArgumentError("Expect an integer");
            }
            ki->kernarg_align = (unsigned)value->GetValue();
        } else if (key->GetValue() == ".kernarg_segment_size") {
            auto value = Object::dyn_cast<Integer>(e.second);
            if (!value) {
                return absl::InvalidArgumentError("Expect an integer");
            }
            ki->kernarg_size = (unsigned)value->GetValue();
        } else if (key->GetValue() == ".private_segment_fixed_size") {
            auto value = Object::dyn_cast<Integer>(e.second);
            if (!value) {
                return absl::InvalidArgumentError("Expect an integer");
            }
            ki->private_segment_fixed_size = (unsigned)value->GetValue();
        }
    }

    if (has_name) {
        return absl::OkStatus();
    }
    return absl::InvalidArgumentError("Invalid metadata for kernel: " + *name);
}

KernelArgument CodeObjectV3MetadataParser::ParseArgument(msgpack::Map *m,
                                                         absl::Status *stat) {
    size_t size = 0;
    size_t offset = 0;
    bool is_pointer = false;
    std::string name;
    std::string value_kind;

    for (const auto &e : m->GetValue()) {
        auto key = Object::dyn_cast<String>(e.first);
        if (key->GetValue() == ".name") {
            if (auto v = Object::dyn_cast<String>(e.second)) {
                name = std::string(v->GetValue());
            } else {
                *stat = absl::InvalidArgumentError("No argument name");
                return KernelArgument::Invalid();
            }
        } else if (key->GetValue() == ".type_name") {
            if (auto v = Object::dyn_cast<String>(e.second)) {
                auto u = v->GetValue();
                is_pointer |= u.size() && u.back() == '*';
            } else {
                *stat = absl::InvalidArgumentError(
                    "Expect a string as argument type name");
                return KernelArgument::Invalid();
            }
        } else if (key->GetValue() == ".value_kind") {
            if (auto v = Object::dyn_cast<String>(e.second)) {
                // HACK: record the name as the name of the argument
                auto u = v->GetValue();
                auto w = std::string(u);
                static const std::set<std::string> kImplicitArguments{{
                    "global_buffer",
                    "dynamic_shared_pointer",
                    "queue",
                    "hidden_printf_buffer",
                    "hidden_hostcall_buffer",
                    "hidden_default_queue",
                    "hidden_completion_action",
                    "hidden_multigrid_sync_arg",
                    "hidden_global_offset_x",
                    "hidden_global_offset_y",
                    "hidden_global_offset_z",
                    "hidden_none",
                }};
                if (kImplicitArguments.count(std::string(u))) {
                    is_pointer = true;
                    if (absl::StartsWith(u, "hidden_") && name.empty()) {
                        name = u;
                    }
                }
            } else {
                *stat =
                    absl::InvalidArgumentError("Expect a string as value_kind");
                return KernelArgument::Invalid();
            }
        } else if (key->GetValue() == ".size") {
            if (auto v = Object::dyn_cast<Integer>(e.second)) {
                size = v->GetValue();
            } else {
                *stat = absl::InvalidArgumentError("Expect an integer as size");
                return KernelArgument::Invalid();
            }
        } else if (key->GetValue() == ".offset") {
            if (auto v = Object::dyn_cast<Integer>(e.second)) {
                offset = v->GetValue();
            } else {
                *stat = absl::InvalidArgumentError("Expect an integer as size");
                return KernelArgument::Invalid();
            }
        }
    }
    return KernelArgument(name,
                          is_pointer ? KernelArgument::kArgPointer
                                     : KernelArgument::kArgPrimitive,
                          size, offset);
}

} // namespace ocl::hip
