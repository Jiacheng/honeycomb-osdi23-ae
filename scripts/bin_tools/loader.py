import os
import sys

repo_path = os.path.abspath("..")
source_dir = sys.argv[1]
target_dir = repo_path

for root, dirs, files in os.walk(source_dir):
    for f in files:
        source_path = os.path.join(root, f)
        target_path = target_dir + root.split(source_dir)[1]
        if target_path.find(".git") == -1: 
            print(target_path)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            os.system("cp " + source_path + " " + target_path)
        