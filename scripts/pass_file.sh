#!/bin/sh

password=${PASSWORD:='asdf!1234'}

remote_user="root"
remote_host="127.0.0.1"

sshpass -p "$password" scp -r -P 10022 "$ae_experiment_path" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
sshpass -p "$password" scp -P 10022 "${ae_rocm_lib_path}/libOpenCL.so.1" "${remote_user}@${remote_host}:/lib/x86_64-linux-gnu"
sshpass -p "$password" scp -P 10022 "${ae_runtime_build_path}/userspace/lib/opencl/hip/libamdhip64.so" "${remote_user}@${remote_host}:/lib/x86_64-linux-gnu"
sshpass -p "$password" scp -P 10022 "${ae_runtime_build_path}/userspace/lib/opencl/hip/libamdhip64.so.5" "${remote_user}@${remote_host}:/lib/x86_64-linux-gnu"
sshpass -p "$password" scp -P 10022 "${ae_runtime_build_path}/userspace/lib/opencl/cratercl/libcratercl.so" "${remote_user}@${remote_host}:${ae_vm_test_folder}/library_home"
sshpass -p "$password" scp -P 10022 "${ae_accel_build_path}/bin_tools/data.tar.gz" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
sshpass -p "$password" scp -r -P 10022 "${CL_CACHE_DIR}" "${remote_user}@${remote_host}:${ae_vm_test_folder}/binary_home"
sshpass -p "$password" scp -P 10022 "${ae_scripts_path}/figure7/run_honeycomb_runtime_baseline.sh" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
sshpass -p "$password" scp -P 10022 "${ae_scripts_path}/figure7/run_honeycomb_runtime_SM.sh" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
sshpass -p "$password" scp -P 10022 "${ae_scripts_path}/figure7/run_honeycomb_runtime_SM_MEM.sh" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
sshpass -p "$password" scp -P 10022 "${ae_scripts_path}/figure7/run_honeycomb_runtime_SM_MEM_V.sh" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
sshpass -p "$password" scp -P 10022 "${ae_scripts_path}/figure7/input_command" "${remote_user}@${remote_host}:${ae_vm_test_folder}"
