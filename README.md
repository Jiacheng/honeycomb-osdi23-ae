
# 1. Summary

Honeycomb is a software-based, secure, and efficient TEE for GPU computations. The key ideas of Honeycomb are:

1. Leverage program analysis techniques to validate that for each GPU kernel all of its memory accesses to the GPU memory stay within bounds and conform with the security policy.
2. Intercept the interactions between the host and the GPU using a security monitor so that:
   i. the GPU only executes kernels that have been validated.
   ii. the confidentiality and integrity of the data for each application are maintained.

Our preliminary evaluation shows that the approach incurs small performance overheads for real world applications like SpecACCEL and the inference workload of the ResNet neural network.

This package provides the artifacts described in the paper, including:

1. The validator (available in source code). It parses the GPU binaries that target the AMD RDNA2 ISA and validates whether the GPU kernels in the binaries are safe, i.e., conforming with the security policy. 
We include the source code of the validator in this package. To ease the evaluation of the validator we make it a userspace executable.
2. The virtual machines and the security monitor (available as an Internet-accessible machine). It consists of the security monitors for the GPU and applications, as well as various infrastructures (e.g., patched kernels and qemu) to deploy them. Note that the current prototype only supports the AMD RX6900XT GPU, and AMD CPU that comes with SEV-SNP support. We provide login credentials to a machine that is equipped with the required hardware and software infrastructures. 
3. The test suites (available in source code and patches). The package consists of a benchmark of the ResNet18 inference workload (available in source code), as well as our patches to the SpecACCEL benchmark to make it pass the validation. Due to the license of the SpecACCEL benchmark we are only able to make the patches available but not the source code of the benchmark.

We expect the evaluation of the artifacts is able to:

1. Reproduce that the validator validates the GPU kernels from the SpecACCEL benchmark suites (except 128.heartwall and 140.bplustree, as noted in the submission) and the ResNet inference benchmark.
2. Successfully run and retrieve the performance numbers for the benchmarks described in the paper.
3. Run all the benchmarks inside the GPU TEE and demonstrate the security of the system.

**If you are evaluating on the machine we provided you can skip directly to Section 4.** 

# 2. Prerequisites

The evaluation requires requires the following software and hardware components to function properly:

1. An AMD, Zen-3 CPU that comes with SEV-SNP support.
2. An AMD 6900XT GPU.
3. A patched Linux kernel that allows the machine to serve as a SEV-SNP host, and allows passing the GPU into virtual machines.
4. Toolchains to build Honeycomb from source. We build the current prototype of Honeycomb using gcc 11, Rust 1.65.0 and Rust nightly on top of Ubuntu 22.04.
5. Runtime support to run GPU applications on top of the AMD 6900XT GPU, including:
  * ROCm 5.4.0 to run the benchmarks to retrieve the baseline numbers.
  * Custom OpenCL / HIP runtime to interact with Honeycomb, both on the host and the guest.
5. Various components to launch SEV-SNP guests, including:
  * Patched QEMU and OVMF that support launching SEV-SNP guest virtual machines.
  * A Limine v4 bootloader to bootstrap Honeycomb

To facilitate the evaluation we set up the evaluation environment on an Internet-accessible machine. The login credentials will be provided upon requests.

# 3. Building Honeycomb and the benchmarks

To facilitate evaluation we build the validator as a userspace binary. You can run the following commands to build the validator:

```shell
$ cd $HONEYCOMB_REPO_DIR/hypervisor
$ cargo build --release -p validator
```

You can run the following commands to build the micro benchmarks and the ResNet18 deep learning inference benchmark using CMake: 

```shell
cd $HONEYCOMB_REPO_DIR
mkdir build
cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm ../
cd ..
cmake --build ./build
```

# 4. Reproducing the results

We consider two components when evaluating the artifacts. First, you can run the validator in user space to reproduce the results of static analysis over the SpecACCEL benchmark suites as well as the ResNet18 inference workload (Figure 9). The validator takes two inputs, the ELF binary and a JSON file that describes the pre-conditions of the validations. It checks all load and store instructions of the described kernels to ensure that all access conform to the security policy.

Second, we set up the machine to reproduce results on running validated GPU applications inside TEE. Several implementation caveats, however, are worth noting. Honeycomb utilizes PCIe passthrough to protect the GPU from other processes running on the host machine. The GPU is exclusively bound to either the host or the Honeycomb security monitor (SM) during the evaluation. The GPU is bound to the host machine when collecting the baseline numbers and is bound to the SM otherwise. The script `disable-gpu-passthrough.sh` and `enable-gpu-passthrough.sh` bind the GPU to the host and the SM respectively. Note that the machine might hang due to the complexity of PCIe passthrough -- rebooting the machine will solve the problem.

The current artifact does not support running the evaluations of TEE VM concurrently.

## Sanity checks: validating trivial programs using the validator 

We provide several examples in `hypervisor/validator/testcases/loops` to ensure that the validator is working correctly. You can compile the test case to ELF using the clang. For example: 

```shell
$ /opt/rocm/llvm/bin/clang -x hip --cuda-device-only --no-gpu-bundle-output -O2 --offload-arch=gfx1030 matrix.cu -o matrix.bin
```

Compile the test case of `matrix.cu` to the GPU binary. The validator can now validate the binary with the corresponding preconditions:

```shell
$ $HONEYCOMB_REPO_DIR/hypervisor/target/release/validator matrix.bin matrix.json
```

The validator outputs `0 remarks` indicating that all load / store instructions conform to the security policy.

## Figure 6: Bandwidth of secure memory copy of Honeycomb

We collect the numbers of the `memcpy` micro benchmark on the host machine. Here are the commands to reproduce results of Figure 6:

```shell
export GPUMPC_RESOURCE_DIR=$HONEYCOMB_REPO_DIR/build/experiments/
$HONEYCOMB_REPO_DIR/experiments/memcpy/run.sh
```

## Figure 7: Overall performance of SPEC ACCEL and ResNet18 on Honeycomb

The script `/data/honeycomb-ae/opt/honeycomb/evaluation/performance/figure7-data.sh` executes each benchmark and records the wall clock time to finish each of them. You can collect the data point by running:

```shell
$ cd /data/honeycomb-ae/opt/honeycomb/evaluation/performance
$ ./figure7-data.sh --series <SERIES>
```

Where the series can be `baseline`, `driver`, `sm`, `sm+mem`, or `full`, each of which corresponds to the data series in Figure 7.

Note that the data of both the `baseline` and `drivers` series should be collected on the host machine, while the other three are collected running inside the application VM.

To collect data in the host, please follow the steps below to setup the environment

1. Since there is only one GPU and there are some hard encoded paths, two users can not evaluate the artifact at the same time. Please check if others are doing the evaluation using `who` and `ps aux | grep qemu`. If there is none, you could safely reboot the machine using `sudo bash /data/honeycomb-ae/opt/honeycomb/bin/reboot.sh` to get a clean environment. This is due to the limitation of the vfio-pci driver where GPU could not be bound back to the host again and a hard reboot is needed.
2. Run `sudo bash /data/honeycomb-ae/opt/honeycomb/bin/enable-gpu-host.sh` to make host work
3. Start the benchmark and collect the data in application VM. For example, run `figure7-data.sh --series driver`.

To collect data inside the application VM, please follow the steps below to set up the environment:

1. Run `sudo bash /data/honeycomb-ae/opt/honeycomb/bin/enable-gpu-passthrough.sh` so that the GPU can be bound to the dom0 VM. This should only be done once.
2. Spawn the dom0 VM by running the following command. It is expected to have no output.

```
$ cd /data/honeycomb-ae/opt/honeycomb/dom0-vm/
$ ./run-dom0-vm.sh
```

3. Start a new shell and SSH into the dom0 via `ssh -p 10022 root@127.0.0.1`. The password is `honeycomb`.
4. In the dom0 console, insert the amdgpu driver and start the userspace host agent in dom0. It is expected to have no output.

```shell
$ modprobe amdgpu
$ /data/honeycomb-ae/opt/honeycomb/bin/host-agent -shm_file /dev/kvmfr0 -vram_size 16368
```

5. Start a new shell and start a minicom console for application VM. You might press enter multiple times after app-vm is launched to see the output.

```shell
$ cd /data/honeycomb-ae/opt/honeycomb/app-vm
$ minicom -Dunix#./console.sock
```

6. Start a new shell and spawn the application VM:

```shell
$ cd /data/honeycomb-ae/opt/honeycomb/app-vm
$ ./run-app-vm.sh
```

There are chances that the console will stuck at freeing memory without getting the IP addresses. You might need to kill the script and try a few times. 

7. Go back to the minicom console, then start the benchmark and collect the data in application VM. For example, run `figure7-data.sh --series sm`. Note that in the VM if you killed the running script (i.e. the program exited abnormally) and tries to run another program, you need to restart the host-agent, otherwise it will crash and your next program will stuck. This is a known limitation.

## Figure 9: Latency on validating GPU kernels

On the host machine, run the following script to reproduce Figure 9:
```shell
$HONEYCOMB_REPO_DIR/scripts/figure9/reproduce_figure_9.sh
```

# Contacts 

Points of contacts for artifacts evaluation:

* [Haohui Mai](mailto:haohui.mai@gmail.com) 
* [Jiacheng Zhao](mailto:zhaojiacheng@ict.ac.cn)

# Known issues

* The current artifacts have significant degraded performances with respect to the numbers of Figure 7. We are actively investigating the issue. 

# Appendix: Reproducing the environments

This section describes the steps to reproduce the environment for artifact evaluation. Note that the environment has been set up on the machine we provided for artifact evaluation. 

## Installing Honeycomb 

Run the following commands under the build directory to install required files into the directory `<HONEYCOMB_PREFIX>`:

```shell
$ cmake --build . --target INSTALL --prefix <HONEYCOMB_PREFIX>
```

## Patching the SpecACCEL benchmark suites

We install the SpecACCEL benchmarks into different directories for each variants. Following the guideline of in the original distribution, you can install the SpecACCEL 1.2 benchmark suites into the `<ACCEL_PREFIX>` directory using the following command:

```shell
$ ./install.sh -u linux-suse10-amd64 -d <ACCEL_PREFIX>
```

We keep our changes of SpecACCEL in the `accel` repository. The `baseline-rocm` and `runtime-checks` branchs contain changes to run on modern ROCm platforms and required changes to pass validations. You can use `rsync` to path the installation:

```shell
$ cd <REPO_OF_ACCEL>
$ rsync -av * <ACCEL_PREFIX>
$ find <ACCEL_PREFIX>/benchspec/ACCEL/ -name "parboil.h"|xargs rm
```

Note that the utility scripts assume that the two installations reside in `<HONEYCOMB_AE_ROOT>/accel/baseline` and `<HONEYCOMB_AE_ROOT>/accel/validated` respectively. Now you can use `runspec` to run the benchmark suites:

```shell
$ . <ACCEL_PREFIX>/shrc 
$ runspec --config opencl-amd-rocm.cfg --noreportable --platform amd --device GPU --rebuild opencl --noreportable --tune=base --iterations=1 --action setup
$ runspec --config opencl-amd-rocm.cfg --noreportable --platform amd --device GPU --rebuild opencl --noreportable --tune=base --iterations=1 --action run 
```

Note that the application VM we used for artifact evaluation has insufficient dependency to execute the `runspec` script. The scripts we provide to reproduce Figure 7 only record the performance numbers but do not checks the results like what `runspec` does. 
