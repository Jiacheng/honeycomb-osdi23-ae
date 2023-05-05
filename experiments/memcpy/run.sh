#!/usr/bin/env bash
if [ -z "$GPUMPC_RESOURCE_DIR" ]; then
    echo "Must set env var GPUMPC_RESOURCE_DIR"
    exit 1
fi

unset LD_PRELOAD

warmup=5
loop=100

echo "Running benchmark for original hip runtime..."
$GPUMPC_RESOURCE_DIR/memcpy/memcpy_benchmark -warmup $warmup -loop $loop
#new line
echo

echo "Running benchmark for hip-hack..."
LD_PRELOAD=$GPUMPC_RESOURCE_DIR/../userspace/lib/opencl/usm/libhip-hack.so \
    $GPUMPC_RESOURCE_DIR/memcpy/memcpy_benchmark -warmup $warmup -loop $loop
