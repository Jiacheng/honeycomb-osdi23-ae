# ROCm-based unit tests

This directory contains unit tests that invokes the low-level APIs of ROCR-Runtime / ROCT-Thunk-Interface to interact with the GPU.

The tests are disabled by default. Specify `ROCR_RUNTIME_SRC_DIR` to enable the builds. Note that it requires a static build of ROCR-Runtime to work.

