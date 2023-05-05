#!/bin/sh
python merge.py
gnuplot -c benchmark-app.gnuplot ${CSV_OUTPUT_DIR}/merged.csv