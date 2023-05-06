#include "host_agent.h"
#include "host_environment.h"
#include "opencl/hsa/enclave/idl.h"

#include <atomic>
#include <gflags/gflags.h>

DEFINE_string(shm_file, "", "The filename of the shared memory");
DEFINE_uint64(gtt_size, 64, "The size of the reserved GTT memory in MB");
DEFINE_uint64(gtt_vaddr, 1ull << 30,
              "The virtual address of the GTT address space");
DEFINE_uint64(vram_size, 1024, "The size of the reserved VRAM memory in MB");
DEFINE_uint64(vram_vaddr, 8ull << 30,
              "The virtual address of the VRAM address space");
DEFINE_int32(
    map_remote_pfn, 0,
    "allows to map physical pages of the host into its own address space");

using namespace ocl::hsa;
using namespace ocl::hsa::enclave;

int main(int argc, char *argv[]) {
    using idl::ConfigurationSpaceLayout;
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Platform &plat = Platform::Instance();
    auto stat = plat.Initialize();
    if (!stat.ok() || plat.GetDevices().empty()) {
        std::cerr << "Failed to initialize the GPU platform\n";
        return -1;
    }

    auto dev = plat.GetDevices()[0];

    HostEnvironment::Options options = {
        .gtt_vaddr = FLAGS_gtt_vaddr,
        .gtt_size = FLAGS_gtt_size << 20,
        .vram_vaddr = FLAGS_vram_vaddr,
        .vram_size = FLAGS_vram_size << 20,
        .map_remote_pfn = !!FLAGS_map_remote_pfn,
    };
    HostEnvironment env;
    stat = env.Open(FLAGS_shm_file, options, dev);
    if (!stat.ok()) {
        std::cerr << "Failed to initialize the host agent environment: "
                  << stat.ToString() << "\n";
        return -1;
    }

    char *conf_space =
        reinterpret_cast<char *>(env.GetConfigurationSpaceBase());
    auto tx_watermark = reinterpret_cast<ConfigurationSpaceLayout::Watermark *>(
        conf_space + ConfigurationSpaceLayout::kTXBufferWatermarkOffset);
    auto rx_watermark = reinterpret_cast<ConfigurationSpaceLayout::Watermark *>(
        conf_space + ConfigurationSpaceLayout::kRXBufferWatermarkOffset);

    TransmitBuffer tx(
        absl::MakeSpan(conf_space + ConfigurationSpaceLayout::kTXBufferOffset,
                       ConfigurationSpaceLayout::kTransmitBufferSize),
        reinterpret_cast<std::atomic_size_t *>(&tx_watermark->rptr),
        reinterpret_cast<std::atomic_size_t *>(&tx_watermark->wptr));

    TransmitBuffer rx(
        absl::MakeSpan(conf_space + ConfigurationSpaceLayout::kRXBufferOffset,
                       ConfigurationSpaceLayout::kTransmitBufferSize),
        reinterpret_cast<std::atomic_size_t *>(&rx_watermark->rptr),
        reinterpret_cast<std::atomic_size_t *>(&rx_watermark->wptr));

    HostRequestHandler handler(dev, &tx, &rx);
    stat = handler.Initialize();
    if (!stat.ok()) {
        std::cerr << "Failed to initialize the request handler: "
                  << stat.ToString() << "\n";
        return -1;
    }

    while (true) {
        handler.ProcessRequests();
    }

    return 0;
}
