import os
repo_path = os.path.abspath("..")
source_dir = repo_path + "/benchspec/ACCEL"
target_dir = os.path.abspath("./ACCEL")

def copy_dir(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    os.system("cp -r " + src_path + " " + dst_path)

def export(benchmark):
    run_base_dir = source_dir + "/" + benchmark + "/run"
    run_dst_dir = target_dir + "/" + benchmark
    copy_dir(run_base_dir, run_dst_dir)

config = open("benchmarks.cfg", "r")
benchmarks = list(map(lambda x : x.strip(), config.readlines()))
for benchmark in benchmarks:
    export(benchmark)