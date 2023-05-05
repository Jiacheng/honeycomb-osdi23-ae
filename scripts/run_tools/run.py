import os
import sys

config = open("benchmarks.cfg", "r")
benchmarks = list(map(lambda x : x.strip(), config.readlines()))

repo_path = os.path.abspath("..")
target_dir = repo_path + "/benchspec/ACCEL/"
csv_output_dir = os.environ["CSV_OUTPUT_DIR"]
result_log = open(csv_output_dir + "/rocm_baseline_accel.csv", "w")
print(csv_output_dir + "/rocm_baseline_accel.csv")

# environment
# repo_path = "/data/ghostfly/vm-test/gpumpc"
# os.environ['GPUMPC_RESOURCE_DIR'] = repo_path + "/build/experiments/"
# os.environ['LD_LIBRARY_PATH'] = repo_path + "/build/userspace/lib/opencl/hip"
# os.environ['GPUMPC_STRICT_LAYOUT'] = "1"
# os.environ['GPUMPC_SECURE_MEMCPY'] = "1"

def test_benchmark(benchmark):
    build_path = target_dir + benchmark + "/run/"
    for dir in os.listdir(build_path):
        if(dir.find("run") != -1):
            run_path = os.path.join(build_path, dir) + "/"
            print(run_path)
            exe_name = benchmark.split(".")[1]
            command = open(run_path + "control", "r")
            command_line = command.read()
            command_line = command_line.split(" ", 2)[2]
            print("command line:" + command_line)
            for file in os.listdir(run_path):
                if(file.find("compsys") != -1):
                    print("execute command: cd " + run_path + " && ./" + file + " " + command_line)
                    # print the result to the console
                    # os.system("cd " + run_path + " && " + "./" + file + " " + command_line)
                    # collect all result and only save running time of all benchmarks in the file.
                    result = os.popen("cd " + run_path + " && " + "./" + file + " " + command_line).read()
                    if(result.find("Timer Wall Time: ") != -1):
                        time = result[result.find("Timer Wall Time: "):]
                        result_log.write(benchmark[benchmark.find(".")+1:] + "," + time)

for benchmark in benchmarks: 
    build_path = target_dir + benchmark + "/run/"
    if sys.argv[1] == "clean":
        if os.path.exists(build_path):
            os.system("rm -rf " + build_path + "*")
    elif sys.argv[1] == "run":
        test_benchmark(benchmark)
