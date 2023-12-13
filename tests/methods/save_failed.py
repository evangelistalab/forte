import yaml
import shutil
import os

# Load the yaml file
with open("failed_tests.yaml", "r") as file:
    data = yaml.safe_load(file)

# Iterate through the yaml data
for _, v in data.items():
    for _, dir_list in v.items():
        # Iterate through each directory list
        for dir_name in dir_list:
            # Ensure the directory name is not None and it starts with '-'
            if dir_name:
                dir_name = dir_name.strip() # strip whitespaces
                
                # Create file paths
                src_path = os.path.join(dir_name, "output.dat")
                dest_path = os.path.join(dir_name, "output.failed")
                
                # Check if "output.dat" file exists in the directory
                if os.path.isfile(src_path):
                    # Copy the file
                    shutil.copy(src_path, dest_path)
                    print(f"Copying {src_path} to {dest_path}")
                else:
                    print(f"No output.dat found in {dir_name}. Skipping...")

