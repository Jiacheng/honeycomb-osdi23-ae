#!/usr/bin/env bash
if [ -z "$GPUMPC_RESOURCE_DIR" ]; then
    echo "Must set env var GPUMPC_RESOURCE_DIR"
    exit 1
fi

unset LD_LIBRARY_PATH

warmup=5
loop=100

echo "Running benchmark for original hip runtime..."
$GPUMPC_RESOURCE_DIR/memcpy/memcpy_benchmark -warmup $warmup -loop $loop
#new line
echo

echo "Running benchmark for our runtime..."
LD_LIBRARY_PATH=$GPUMPC_RESOURCE_DIR/../userspace/lib/opencl/hip GPUMPC_SECURE_MEMCPY=1 \
    $GPUMPC_RESOURCE_DIR/memcpy/memcpy_benchmark -warmup $warmup -loop $loop
