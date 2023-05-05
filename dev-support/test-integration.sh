#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
VALIDATOR_DIR="${ROOT_DIR}/hypervisor/validator"
BUILD_DIR="${ROOT_DIR}/build"
READ_ELF="${ROOT_DIR}/hypervisor/target/release/readelf"
OBJ_DUMP="${ROOT_DIR}/hypervisor/target/release/objdump"


function build_bins {
    mkdir -p ${BUILD_DIR}
    pushd ${BUILD_DIR} > /dev/null
    cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm ..
    cmake --build .
    if [ $? -ne 0 ]; then
        echo "Build userspace libraries and experiments failed"
        exit 1
    fi
    popd > /dev/null
}

function build_validator {
    pushd ${VALIDATOR_DIR} > /dev/null
    RUSTFLAGS="--deny warnings" cargo build --release -p validator -q
    if [ $? -ne 0 ]; then
        echo "Build validator failed"
        exit 1
    fi
    popd > /dev/null
}

function test_objdump {
    TMP0=`mktemp`
    TMP1=`mktemp`
    TMP2=`mktemp`
    # remove debug symbols
    llvm-strip-15 -x $1 -o $TMP0
    llvm-objdump-15 -dz --no-leading-addr $TMP0 | sed -r '
        /^<\.text>:$/d;         # remove .text label (which is not a symbol)
        /^<__.*>:$/d;           # remove debug entries (weak symbols)
        /^<.*>:$/,$!d;          # remove anything before the first label
        s/^<(.*)>:$/\1:/;       # format the label
        s/\/\/.*$//;            # remove comments
        /^$/d                   # remove blank lines
        ' > $TMP1
    $OBJ_DUMP $1 > $TMP2
    if [ $? -ne 0 ]; then
        echo "Failed to run objdump at $1"
        exit -1
    fi
    diff -uwb $TMP1 $TMP2
    if [ $? -ne 0 ]; then
        echo "Test on objdump failed at $1"
        exit -1
    fi
    rm $TMP0 $TMP1 $TMP2
}

function test_readelf {
    pushd ${VALIDATOR_DIR} > /dev/null
    $READ_ELF $1 > /dev/null
    if [ $? -ne 0 ]; then
        echo "Test on read-elf failed at $1"
        exit -1
    fi
    popd > /dev/null
}


build_bins
build_validator
files=`find build/experiments -name "*.bin"`
for file in $files; do
    echo "Testing ${file}"
    test_readelf "${ROOT_DIR}/$file"
    test_objdump "${ROOT_DIR}/$file"
done

echo "Test succeed."