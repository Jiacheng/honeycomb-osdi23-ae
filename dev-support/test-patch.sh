#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
VALIDATOR_REL_DIR="hypervisor/validator"
VALIDATOR_DIR="${ROOT_DIR}/${VALIDATOR_REL_DIR}"

function check_validator_build {
    pushd ${VALIDATOR_DIR} > /dev/null
    result=`RUSTFLAGS="--deny warnings" cargo build --release -p validator 2>&1`
    if [ $? -ne 0 ]; then
        echo "Build validator failed"
        printf "%s\n" "${result}"
        exit 1
    fi
    popd > /dev/null
}

function check_validator_format {
    pushd ${VALIDATOR_DIR} > /dev/null
    result=`cargo fmt --check -p validator`
    if [ $? -ne 0 ]; then
        echo "Format error in validator"
        printf "%s\n" "${result}"
        exit 1
    fi
    popd > /dev/null
}

function check_validator_test {
    pushd ${VALIDATOR_DIR} > /dev/null
    result=`cargo test -p validator 2>&1`
    if [ $? -ne 0 ]; then
        echo "Test failed in validator"
        printf "%s\n" "${result}"
        exit 1
    fi
    popd > /dev/null
}

function check_validator_clippy {
    pushd ${VALIDATOR_DIR} > /dev/null
    result=`RUSTFLAGS="--deny warnings" cargo clippy -p validator 2>&1`
    if [ $? -ne 0 ]; then
        echo "Clippy check failed in validator"
        printf "%s\n" "${result}"
        exit 1
    fi
    popd > /dev/null
}

function check_validator {
    check_validator_build
    check_validator_format
    check_validator_test
    check_validator_clippy
}

# Usage:
# 1. test-patch
#       run test patch
# 2. test-patch <commit-sha>
#       test changed files since the given commit

validator_pattern="^${VALIDATOR_REL_DIR}/.*\.(rs|toml)$"
if [ -z "$1" ]; then
    check_validator
else
    validator_diffs=`git diff $1 --name-only | grep -E ${validator_pattern}`
    if test ! -z "$validator_diffs"; then
        check_validator
    fi
fi

echo "Test succeed."
