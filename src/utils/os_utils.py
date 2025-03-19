import os

def check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True