#include <chrono>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <iostream>
#include <sys/ioctl.h>
#include <unistd.h>

DEFINE_int32(warmup, 1000000, "warmup times for benchmark");
DEFINE_int32(loop, 3000000, "loop run times for benchmark");

class G6IPCBenchmark {
  public:
    int Initialize();
    void Close();
    int Loop(int times);
    G6IPCBenchmark() : fd_(0) {}
    ~G6IPCBenchmark();

  private:
    int fd_;
};

int G6IPCBenchmark::Initialize() {
    fd_ = open("/dev/g6", O_RDWR | O_CLOEXEC);
    if (fd_ < 0) {
        return -1;
    }
    return 0;
}

void G6IPCBenchmark::Close() {
    close(fd_);
    fd_ = 0;
}

int G6IPCBenchmark::Loop(int times) {
    for (int i = 0; i < times; i++) {
        int ret = ioctl(fd_, 0, 0);
        if (ret) {
            return ret;
        }
    }
    return 0;
}

G6IPCBenchmark::~G6IPCBenchmark() { Close(); }

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    G6IPCBenchmark benchmark;
    if (benchmark.Initialize()) {
        std::cerr << "Failed to initialize the benchmark\n";
        return -1;
    }

    if (benchmark.Loop(FLAGS_warmup)) {
        std::cerr << "Failed to warm up\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    if (benchmark.Loop(FLAGS_loop)) {
        std::cerr << "Failed to loop\n";
        return -1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double delta =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "Total time " << delta << " ms out of " << FLAGS_loop
              << " times. Average: " << delta / FLAGS_loop << " ms\n";
    benchmark.Close();
    return 0;
}