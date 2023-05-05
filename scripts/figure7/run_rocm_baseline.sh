#!bin/sh

if [ ! -d ${ae_experiment_path}/resnet/output ]; then
  mkdir ${ae_experiment_path}/resnet/output
fi

if [ ! -d ${ae_experiment_path}/resnet/result ]; then
  mkdir ${ae_experiment_path}/resnet/result
fi

cd ${ae_accel_build_path}/run_tools
python run.py run


> "${RESULT_DIR}${ae_experiment_path}/resnet/result/rocm_baseline_resnet.csv"

RESNET_FOLDER="${ae_experiment_path}/resnet"
cd ${RESNET_FOLDER}
./resnet_benchmark -warmup 1000 -loop 1000 > "${ae_experiment_path}/resnet/output/resnet"

grep -r "Total: .*ms" ${ae_experiment_path}/resnet/output > "${ae_experiment_path}/resnet/result/time.txt"

while IFS=':' read -r filepath content; do
  program=$(basename "${filepath}")
  echo "resnet,${content}," >> "${ae_experiment_path}/resnet/result/rocm_baseline_resnet.csv"
done < "${ae_experiment_path}/resnet/result/time.txt"
cp "${ae_experiment_path}/resnet/result/rocm_baseline_resnet.csv" ${CSV_OUTPUT_DIR}