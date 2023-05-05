import os
import sys

repo_path = os.path.abspath("..")
source_dir = repo_path + "/benchspec/ACCEL"

cl_dir = os.path.abspath("./cl")
bin_dir = os.path.abspath("./bin")

runtime = sys.argv[1]
ocl_cc_dir = os.environ['ocl_cc_dir']

def collect(benchmark) -> bool:
    cl_root_dir = source_dir + "/" + benchmark + "/data/all/input"
    find_cl_file = False
    dst_path = cl_dir + "/" + benchmark + "/"
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for root, dirs, files in os.walk(cl_root_dir):
        for f in files:
            if f[-3:] == ".cl":
                find_cl_file = True
                os.system("cp " + os.path.join(root, f) + " " + dst_path)
    print(benchmark, find_cl_file)
    if find_cl_file:
        src_dir = source_dir + "/" + benchmark + "/src/*"
        os.system("cp -r " + src_dir + " " + dst_path)
    return find_cl_file

def generate_binary(benchmark):
    cl_path = cl_dir + "/" + benchmark
    bin_path = bin_dir + "/" + benchmark
    if not os.path.exists(bin_path):
        os.makedirs(bin_path)
    for root, dirs, files in os.walk(cl_path):
        for f in files:
            if f[-3:] == ".cl":
                cl_file_path = os.path.join(root, f)
                i_path = cl_file_path[:-3] + ".i"
                print(cl_file_path)
                print(i_path)
                if benchmark.find("116") != -1:
                    os.system("clang -E " + cl_file_path + " -o " + i_path + " -DBLOCK_SIZE=16 -D PRESCAN_THREADS=512 -D KB=64 -D UNROLL=16 -D BINS_PER_BLOCK=65536 -D BLOCK_X=14")
                elif benchmark.find("127") != -1 and runtime == "gpumpc":
                    os.system("clang -E " + cl_file_path + " -o " + i_path + " -DBLOCK_SIZE=16 -D MAX_ROW=4096")
                else:
                    os.system("clang -E " + cl_file_path + " -o " + i_path + " -DBLOCK_SIZE=16")
                #os.system("./ocl-cc -o " + bin_path + "/" + f[:-3] + ".bin " + i_path)
                os.system(ocl_cc_dir + " -o " + bin_path + "/" + f[:-3] + ".bin " + i_path)

def paste(benchmark):
    dst_base_dir = source_dir + "/" + benchmark + "/data/all/input"
    print("dst:" + dst_base_dir)
    for root, dirs, files in os.walk(dst_base_dir):
        for f in files:
            if f[-3:] == ".cl":
                src_path = bin_dir + "/" + benchmark + "/" + f[:-3] + ".bin"
                dst_path = root + "/" + f[:-3] + ".bin"
                os.system("cp " + src_path  + " " + dst_path)
                print("cp " + src_path  + " " + dst_path)

def workflow(benchmark):
    if collect(benchmark):
        generate_binary(benchmark)
        paste(benchmark)
config = open("benchmarks.cfg", "r")
benchmarks = list(map(lambda x : x.strip(), config.readlines()))
for benchmark in benchmarks:
    workflow(benchmark)