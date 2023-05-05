#!/bin/sh

modprobe amdgpu
export CL_CACHE_DIR=${CL_CACHE_DIR}
export GPUMPC_RESOURCE_DIR="/test/experiments"

TEST_FOLDER=$(test_folder)

SCRIPT_DIR=$(pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output"
RESULT_DIR="${SCRIPT_DIR}/result"

if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir "${OUTPUT_DIR}"
fi

if [ ! -d "${RESULT_DIR}" ]; then
  mkdir "${RESULT_DIR}"
fi

> "${RESULT_DIR}/ACCEL_baseline_output.csv"

ACCEL_FOLDER=$(TEST_FOLDER)/ACCEL
if [ ! -d "${ACCEL_FOLDER}" ]; then
  tar -xzvf data.tar.gz
fi

cd ${ACCEL_FOLDER}

while read line1; do
    $line1
    read line2;
    output_file=$(echo $(basename ${line2%% *}) | sed "s/_exe_base.compsys//")
    $line2 > "${OUTPUT_DIR}/${output_file}"
done < "${SCRIPT_DIR}/input_command"

grep -r "Timer Wall Time: .*" ${OUTPUT_DIR} > "${RESULT_DIR}/time.txt"

while IFS=':' read -r filepath content; do
  program=$(basename "${filepath}")
  echo "${program},${content}," >> "${RESULT_DIR}/ACCEL_baseline_output.csv"
done < "${RESULT_DIR}/time.txt"

> "${RESULT_DIR}/Resnet18_baseline_output.csv"

RESNET_FOLDER="$TEST_FOLDER/experiments/resnet"
echo "$RESNET_FOLDER"
cd ${RESNET_FOLDER}
./resnet_benchmark -warmup 1000 -loop 1000 > "${OUTPUT_DIR}/Resnet18"

grep -r "Total: .*ms" ${OUTPUT_DIR} > "${RESULT_DIR}/time.txt"

while IFS=':' read -r filepath content; do
  program=$(basename "${filepath}")
  echo "resnet,${content}," >> "${RESULT_DIR}/Resnet18_baseline_output.csv"
done < "${RESULT_DIR}/time.txt"