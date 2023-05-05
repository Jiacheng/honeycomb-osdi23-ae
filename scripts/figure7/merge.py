import csv
import re
import os

csv_output_dir = os.environ["CSV_OUTPUT_DIR"]
files = ["rocm_baseline_accel.csv", "rocm_baseline_resnet.csv", "ACCEL_baseline_output.csv", "Resnet18_baseline_output.csv", "ACCEL_SM_output.csv", "Resnet18_SM_output.csv", "ACCEL_SM_MEM_output.csv", "Resnet18_SM_MEM_output.csv", "ACCEL_SM_MEM_V_output.csv", "Resnet18_SM_MEM_V_output.csv"]

data = {}

number_regex = re.compile(r"[-+]?\d*\.\d+|\d+")

for file in files:
    with open(csv_output_dir + "/" + file, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name = row[0]
            value = row[1]
            
            numbers = number_regex.findall(value)
            
            if name in data:
                data[name].extend(numbers)
            else:
                data[name] = numbers

output_file = csv_output_dir + "/merged.csv"
with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["# Workload", "Baseline", "runtime", "SM", "SH+Mem", "SH+Mem+Validation"])
    for name, values in data.items():
        writer.writerow([name] + values)