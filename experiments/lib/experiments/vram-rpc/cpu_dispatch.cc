#include "cpu_dispatch.h"
#include "utils/hip_helper.h"
#include "utils/monad_runner.h"
#include <fstream>
#include <numeric>

namespace gpumpc::experiment {

using namespace ocl::hip;

absl::Status CPURPC::Initialize(const Options &options) {
    if (payload_size_ == 0) {
        return absl::InvalidArgumentError("payload length not initialized");
    }
    if (payload_size_ % kAESBlockSize != 0) {
        return absl::InvalidArgumentError("payload length should be aligned");
    }
    shm_size_ = kPayloadOffset + payload_size_;
    buf_.resize(payload_size_);
    memset(buf_.data(), 0, payload_size_);

    options_ = options;
    auto stat = LoadBinary();
    if (!stat.ok()) {
        return stat;
    }

    auto hipError = hipMalloc(&base_, kDeviceBaseSize);
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("cannot allocate mem");
    }
    hipError = hipMalloc(&request_, payload_size_);
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("cannot allocate mem");
    }

    if (options_.mode == Mode::kServer) {
        stat = InitializeServer();
    } else {
        stat = InitializeClient();
    }

    if (!stat.ok()) {
        return stat;
    }

    void *ptr = mmap(nullptr, shm_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                     shm_fd_, 0);
    if (ptr == MAP_FAILED) {
        return absl::InvalidArgumentError("Cannot mmap");
    }
    shm_ = reinterpret_cast<unsigned char *>(ptr);
    memset(shm_, 0, shm_size_);
    client_signal_ =
        reinterpret_cast<std::atomic_ulong *>(shm_ + kClientSignalOffset);
    server_signal_ =
        reinterpret_cast<std::atomic_ulong *>(shm_ + kServerSignalOffset);
    timestamp_shm_ = shm_ + kTimestampOffset;
    payload_ = shm_ + kPayloadOffset;

    // TODO: Diffe-Hellman
    const unsigned kMagic = 0xdeadbeaf;
    const unsigned ukey[kAES256KeySizeInWord] = {
        kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic, kMagic};
    const unsigned iv[kAESIVSizeInWord] = {kMagic, kMagic, 0, 0};
    memcpy(ukey_, ukey, sizeof(ukey));
    memcpy(iv_, iv, sizeof(iv));

    // openssl context
    if (!(ctx_ = EVP_CIPHER_CTX_new())) {
        return absl::InvalidArgumentError("can not new openssl context");
    }
    if (1 != EVP_EncryptInit_ex(ctx_, EVP_aes_256_ctr(), NULL,
                                reinterpret_cast<const unsigned char *>(ukey_),
                                reinterpret_cast<const unsigned char *>(iv_))) {
        return absl::InvalidArgumentError("can not init openssl");
    }

    return absl::OkStatus();
}

absl::Status CPURPC::Destroy() {
    if (base_) {
        auto hipError = hipFree(base_);
        if (hipError != hipSuccess) {
            return absl::InvalidArgumentError("cannot free mem");
        }
    }
    if (request_) {
        auto hipError = hipFree(request_);
        if (hipError != hipSuccess) {
            return absl::InvalidArgumentError("cannot free mem");
        }
    }
    if (shm_ != MAP_FAILED) {
        munmap(shm_, shm_size_);
    }
    if (shm_fd_) {
        close(shm_fd_);
    }
    return absl::OkStatus();
}

CPURPC::~CPURPC() {
    auto stat = Destroy();
    (void)stat;
}

absl::Status CPURPC::LoadModule(hipModule_t *module, std::string filepath) {
    std::ifstream elf_file(filepath, std::ios::binary);
    if (!elf_file.is_open()) {
        return absl::InvalidArgumentError("can not open file");
    }

    std::string str((std::istreambuf_iterator<char>(elf_file)),
                    std::istreambuf_iterator<char>());
    elf_file.close();

    auto hipError = hipModuleLoadData(module, str.c_str());
    if (hipError != hipSuccess) {
        return absl::InvalidArgumentError("can not load data");
    }
    return absl::OkStatus();
}

absl::Status CPURPC::LoadBinary() {
    auto stat = LoadModule(&module_, options_.module_path);
    if (!stat.ok()) {
        return absl::InvalidArgumentError("Cannot load kernel");
    }

    static const char *f[] = {"PingServerCpu", "PingClientCpu",
                              "PingClientCpuCollect", "GetClock"};
    gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
    runner.Run([&]() { return hipModuleGetFunction(&server_, module_, f[0]); })
        .Run([&]() { return hipModuleGetFunction(&client_, module_, f[1]); })
        .Run([&]() {
            return hipModuleGetFunction(&client_collect_, module_, f[2]);
        })
        .Run(
            [&]() { return hipModuleGetFunction(&get_clock_, module_, f[3]); });

    auto err = runner.code();
    if (err != hipSuccess) {
        return absl::InvalidArgumentError("Cannot load function");
    }
    return absl::OkStatus();
}

absl::Status CPURPC::CalibrateTime() {
    using clock = std::chrono::high_resolution_clock;
    enum {
        kMeasureTime = 100,
        kIntervalMs = 10,
    };

    struct {
        clock::time_point tp;
        unsigned long tsc;
    } measurements[kMeasureTime];

    void *buf = base_;
    for (int i = 0; i < kMeasureTime; i++) {
        measurements[i].tp = std::chrono::high_resolution_clock::now();
        auto stat = LaunchKernel(get_clock_, 1, 1, buf);
        if (!stat.ok()) {
            return stat;
        }

        gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
        runner.Run([&]() { return hipDeviceSynchronize(); }).Run([&]() {
            return hipMemcpyDtoH(&measurements[i].tsc, buf,
                                 sizeof(unsigned long));
        });
        usleep(kIntervalMs * 1000);
        if (runner.code()) {
            return absl::InvalidArgumentError("Cannot calibrate time");
        }
    }
    auto err = hipDeviceSynchronize();
    if (err) {
        return absl::InvalidArgumentError("Cannot calibrate time");
    }

    double r = 0;
    for (int i = 1; i < kMeasureTime; i++) {
        auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         measurements[i].tp - measurements[i - 1].tp)
                         .count();
        auto delta_tsc = measurements[i].tsc - measurements[i - 1].tsc;
        r += (double)delta / delta_tsc;
    }
    ns_per_tsc_ = r / (kMeasureTime - 1);
    printf("Calibrated counter: %.4f ns per GPU clock counter\n", ns_per_tsc_);
    return absl::OkStatus();
}

absl::Status CPURPC::Launch() {
    return options_.mode == Mode::kServer ? LaunchServer() : LaunchClient();
}

absl::Status CPURPC::InitializeServer() {
    shm_fd_ =
        open(options_.ipc_mem_file.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (shm_fd_ < 0) {
        return absl::InvalidArgumentError("Cannot open file");
    }
    if (ftruncate(shm_fd_, shm_size_)) {
        return absl::InvalidArgumentError(
            "Cannot set the size of the shared memory buffer");
    }
    return absl::OkStatus();
}

absl::Status CPURPC::InitializeClient() {
    shm_fd_ = open(options_.ipc_mem_file.c_str(), O_RDWR, 0644);
    if (shm_fd_ < 0) {
        return absl::InvalidArgumentError("Cannot open file");
    }
    return absl::OkStatus();
}

void CPURPC::Signal(std::atomic_ulong *direction) { direction->store(1); }

void CPURPC::Wait(std::atomic_ulong *direction) {
    while (direction->load() == 0) {
    }
    direction->store(0);
}

absl::Status CPURPC::EncryptToPayload() {
    unsigned char buffer[kAESBlockSize];
    memset(buffer, 0, kAESBlockSize);
    memcpy(buffer, &timestamp_, sizeof(unsigned long));
    int updated_len;
    if (1 != EVP_EncryptUpdate(ctx_, timestamp_shm_, &updated_len, buffer,
                               kAESBlockSize)) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    if (updated_len != kAESBlockSize) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    if (1 != EVP_EncryptUpdate(ctx_, payload_, &updated_len,
                               reinterpret_cast<unsigned char *>(buf_.data()),
                               payload_size_)) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    if ((size_t)updated_len != payload_size_) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    return absl::OkStatus();
}

absl::Status CPURPC::DecryptFromPayload() {
    unsigned char buffer[kAESBlockSize];
    int updated_len;
    if (1 != EVP_EncryptUpdate(ctx_, buffer, &updated_len, timestamp_shm_,
                               kAESBlockSize)) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    if (updated_len != kAESBlockSize) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    memcpy(&timestamp_, buffer, sizeof(unsigned long));

    if (1 != EVP_EncryptUpdate(ctx_,
                               reinterpret_cast<unsigned char *>(buf_.data()),
                               &updated_len, payload_, payload_size_)) {
        return absl::InvalidArgumentError("can not encrypt");
    }

    if ((size_t)updated_len != payload_size_) {
        return absl::InvalidArgumentError("can not encrypt");
    }
    return absl::OkStatus();
}

absl::Status CPURPC::GetTimestamp() {
    std::vector<unsigned long> values(1);
    gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
    runner.Run([&]() {
        return hipMemcpyDtoH(values.data(), base_, sizeof(unsigned long));
    });
    if (runner.code() != hipSuccess) {
        return absl::InvalidArgumentError("Cannot get timestamp");
    }
    timestamp_ = values[0];
    return absl::OkStatus();
}

absl::Status CPURPC::FetchRequest() {
    gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
    runner.Run(
        [&]() { return hipMemcpyDtoH(buf_.data(), request_, payload_size_); });
    if (runner.code() != hipSuccess) {
        return absl::InvalidArgumentError("Cannot fetch");
    }
    return absl::OkStatus();
}

absl::Status CPURPC::PutResponse() {
    gpumpc::MonadRunner<hipError_t> runner(hipSuccess);
    runner
        .Run([&]() {
            return hipMemcpyHtoD(request_, buf_.data(), payload_size_);
        });
    if (runner.code() != hipSuccess) {
        return absl::InvalidArgumentError("Cannot put");
    }
    return absl::OkStatus();
}

} // gpumpc::experiment
