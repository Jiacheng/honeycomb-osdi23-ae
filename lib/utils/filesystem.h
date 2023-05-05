#include <absl/status/status.h>
#include <fstream>
#include <vector>

namespace gpumpc {

static inline std::vector<char> ReadAll(const std::string &fn, absl::Status *stat) {
    std::ifstream is(fn, std::ifstream::binary);
    std::vector<char> result;
    char buf[4096];
    while (!is.bad() && !is.eof()) {
        is.read(buf, sizeof(buf));
        size_t count = is.gcount();
        if (!count) {
            break;
        }
        if (!is.bad()) {
            result.insert(result.end(), buf, buf + count);
        }
    }
    if (is.bad()) {
        *stat = absl::InvalidArgumentError("Failed to read the file " + fn);
        return std::vector<char>();
    }
    *stat = absl::OkStatus();
    return result;
}
} // namespace gpumpc