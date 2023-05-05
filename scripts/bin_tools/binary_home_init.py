import os
import hashlib

repo_path = os.path.abspath("..")
source_dir = repo_path + "/benchspec/ACCEL"
config = open("benchmarks.cfg", "r")
benchmarks = list(map(lambda x : x.strip(), config.readlines()))
target_dir = os.environ['CL_CACHE_DIR']
if target_dir == "":
    print("env not set!")
else:
    for benchmark in benchmarks:
        base_dir = source_dir + "/" + benchmark + "/data/all/input"
        new_target_path = target_dir + "/raw_data/" + benchmark
        if not os.path.exists(new_target_path):
            os.makedirs(new_target_path)
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f[-3:] == ".cl" or f[-4:] == ".bin":
                    os.system("cp " + os.path.join(root, f) + " " + new_target_path)

    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f[-3:] == ".cl":
                source_path = os.path.join(root, f)
                binary_path = source_path[:-3] + ".bin"
                with open(source_path, "rb") as source_file:
                    content = source_file.read()
                    sha3_256 = hashlib.sha3_256()
                    sha3_256.update(content)
                    new_file_name = "cl-" + sha3_256.hexdigest() + ".bin"
                    new_file_path = os.path.join(target_dir, new_file_name)
                    os.system("cp " + binary_path + " " + new_file_path)