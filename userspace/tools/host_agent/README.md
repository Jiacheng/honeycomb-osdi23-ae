Enclave guest platform
====

## Overview
The EnclaveGuestPlatform is designed to support GPU workloads inside the SEV-SNP VM via forwarding all GPU-related requests to an agent running on top of the host machine. On a very high level:

* The agent and the guest application communicate via shared memory.
* The guest application maintains identical mappings w.r.t GTT and device memory of the agent so that no address translation is required.
* The guest application writes directly to the command queue but sends a RPC to update the doorbell as the doorbell cannot be easily shared.
* The host agent exposes a UNIX domain socket to emulate interrrupts for the GPUs.

This is the first step towards moving all GPU resources in the SVSM.

## Running

### The host agent

```
userspace/tools/host_agent/host-agent -bind /tmp/host-agent.sock -shm_file /tmp/host-agent.mem
```

The `bind` and `shm_file` arguments specify the path of the UNIX domain socket and the path of the shared memory. QEMU can further takes the file of shared memory and map it into the guest kernel.

More parameters are available to specify the memory layouts, etc. Run `host-agent --help` for more details.

### The guest application

Specifying the paths for the domain socket and the shared memory in the environment variables to run the tests should be sufficient:

```
G6_ENCLAVE_SHM=/tmp/host-agent.mem G66_ENCLAVE_SOCKET=/tmp/host-agent.sock userspace/tests/rocm/sdma_test/enclave_sdma_test
```
