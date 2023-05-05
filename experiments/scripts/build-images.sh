#!/bin/sh

FILES="/lib64/ld-linux-x86-64.so.2 \
/lib/x86_64-linux-gnu/libc.so.6 \
/lib/x86_64-linux-gnu/libdl.so.2 \
/lib/x86_64-linux-gnu/libgcc_s.so.1 \
/lib/x86_64-linux-gnu/libm.so.6 \
/lib/x86_64-linux-gnu/libpthread.so.0 \
/lib/x86_64-linux-gnu/librt.so.1 \
/lib/x86_64-linux-gnu/libz.so.1 \
/lib/x86_64-linux-gnu/liblzma.so.5 \
/lib/x86_64-linux-gnu/libtinfo.so.5 \
/opt/rocm-5.2.0/lib/libhsa-runtime64.so.1 \
/opt/rocm-5.2.0/lib/libamd_comgr.so.2 \
/opt/rocm-5.2.0/hip/lib/libamdhip64.so.5 \
/usr/lib/x86_64-linux-gnu/libdrm_amdgpu.so.1 \
/usr/lib/x86_64-linux-gnu/libdrm.so.2 \
/usr/lib/x86_64-linux-gnu/libelf.so.1 \
/usr/lib/x86_64-linux-gnu/libnuma.so.1 \
/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/usr/lib/x86_64-linux-gnu/libunwind-ptrace.so.0 \
/usr/lib/x86_64-linux-gnu/libunwind.so.8 \
/usr/lib/x86_64-linux-gnu/libunwind-x86_64.so.8 \
/usr/share/libdrm/amdgpu.ids"

if [ ! -d $1 ]; then
    echo "Build an image for evaluation."
    echo "$0 <target directory>"
    exit 1
fi

PARENT=$1
for f in $FILES; do
    mkdir -p $PARENT/$(dirname $f) 
    cp $f $1$f
done

ln -s rocm-5.2.0 $1/opt/rocm
