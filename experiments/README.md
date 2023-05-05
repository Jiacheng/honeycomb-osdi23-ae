# Running the experiments

This document describes how to run the experiments for the S&P 2023 deadline.

## Running the experiments under the Hypervisor

### Building the paravirtualized guest kernel

You need to check out the `g6-sp2023` branch of the linux kernel in the repository, and building a kernel with the configuration specified in `kernels/configs/linux-kernel-config-5.17.0-g6-guest`:

```
git clone -b g6-sp2023 <REPO_BASE_URI>/linux.git
cp kernels/configs/linux-kernel-config-5.17.0-g6-guest linux/.config
cd linux
make oldconfig
make -j
```

The building process will generate the bzImage of the kernel which you will be using in the guest VM. Refer to the README.md in the top directory on how to boot up the guest VM. 

### Building the modified hipamd runtime

The current prototype requires the userspace program to issue a system call to provide the offsets of the ring buffer of the queues (i.e., the doorbells) in order to launch a GPU kernel or to copy the memory. Therefore it requires a modified version of the hipamd runtime.

You need to check out the `g6-sp2023` branch of the HIP, hipamd, ROCclr, ROCm-OpenCL-Runtime, ROCR-Runtime, and the ROCT-Thunk-Interface from the repositories:

```
$ mkdir rocm && cd rocm
$ for i in HIP hipamd ROCclr ROCm-OpenCL-Runtime ROCR-Runtime ROCT-Thunk-Interface; do \
    git clone -b g6-sp2023 <REPO_BASE_URI>/$i.git \
  done
$ cd ..
```

You need to build both the ROCR-Runtime and the hipamd repository. You can follow the instructions of the repositories to build them

```
$ mkdir -p rocm-build/ROCR-Runtime && cd rocm-build/ROCR-Runtime
$ cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DIMAGE_SUPPORT=OFF ../../rocm/ROCR-Runtime/src
$ cmake --build .
$ cd ../.. && mkdir -p rocm-build/hipamd && cd rocm-build/hipamd
$ cmake -G Ninja -DHIP_COMMON_DIR=../../rocm/HIP -DAMD_OPENCL_PATH=../../rocm/ROCm-OpenCL-Runtime _DROCCLR_PATH=../../ROCclr -DCMAKE_PREFIX_PATH="<ROCM_PATH>/" ../../rocm/hipamd
$ cmake --build .
```

You are supposed to have `libamdhip64.so.5` and `libhsa-runtime64.so.1` built and ready in the directories.

### Running the experiments inside the Hypervisor

You can run the `build-images.sh` to build a basic chroot environment to run the experiments. The experiment also requires the resource files to work. You can find the corresponding tarballs in the `experiments/data` directory. You need to specify the environment variable `GPUMPC_RESOURCE_DIR` so that the binaries can find the resource files:

```
GPUMPC_RESOURCE_DIR=... experiments/resnet/resnet_test
```

To enqueue the hardware packets through the hypervisor, you need to specify `LD_LIBRARY_PATH` to force the experiment binaries to load the modified runtime, and to enabble it with `G6_ENABLE_SM=1`:


```
GPUMPC_RESOURCE_DIR=... LD_LIBRARY_PATH=... G6_ENABLE_SM=1 experiments/resnet/resnet_test
```

You should be seeing a line of `G6 SM enabled` which indicates that the security monitor is enabled successfully.