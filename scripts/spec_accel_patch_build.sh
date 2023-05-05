#!bin/sh

HONEYCOMB_DIR="$ae_honeycomb_path"
cd ${HONEYCOMB_DIR}
echo "$HONEYCOMB_DIR"/scripts/bin_tools
cp -r ${HONEYCOMB_DIR}/scripts/bin_tools ${ae_accel_build_path}
cp -r ${HONEYCOMB_DIR}/scripts/run_tools ${ae_accel_build_path}

if [ "$1" = "baseline" ]; then
    PATCH_SRC="$ae_accel_patch_source_baseline_path"
elif [ "$1" = "validator" ]; then
    PATCH_SRC="$ae_accel_patch_source_validator_path"
fi
echo "$PATCH_SRC"
cd ${ae_accel_build_path}/bin_tools
python loader.py ${PATCH_SRC}
echo "finish loading patch source code into ae_accel_build_path."
python generator.py gpumpc
echo "finish generating binary with ocl-cc."
python binary_home_init.py
cd ${ae_accel_build_path}
. ./shrc
runspec --config current.cfg --noreportable --platform amd --rebuild opencl --action runsetup
cd ${ae_accel_build_path}/bin_tools
python exporter.py
tar -czvf data.tar.gz ./ACCEL/
echo "finish package all files and ready for pass files!"