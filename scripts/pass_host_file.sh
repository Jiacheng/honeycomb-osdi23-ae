#!/bin/sh

password=${PASSWORD:='asdf!1234'}

remote_user="root"
remote_host="127.0.0.1"

sshpass -p "$password" scp -P 10023 "${ae_runtime_build_path}/userspace/tools/host_agent/host-agent" "${remote_user}@${remote_host}:/runtime"
sshpass -p "$password" scp -P 10023 "${ae_runtime_build_path}/userspace/lib/opencl/hip/libamdhip64.so" "${remote_user}@${remote_host}:/lib/x86_64-linux-gnu"
sshpass -p "$password" scp -P 10023 "${ae_runtime_build_path}/userspace/lib/opencl/hip/libamdhip64.so.5" "${remote_user}@${remote_host}:/lib/x86_64-linux-gnu"