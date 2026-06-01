import os
import site
import sys

print("LD_LIBRARY_PATH is:", os.environ.get("LD_LIBRARY_PATH"))

def setup_nvidia_paths():
    for site_pkg in site.getsitepackages():
        nvidia_dir = os.path.join(site_pkg, "nvidia")
        if os.path.exists(nvidia_dir):
            paths = []
            for d in os.listdir(nvidia_dir):
                lib_dir = os.path.join(nvidia_dir, d, "lib")
                if os.path.exists(lib_dir):
                    paths.append(lib_dir)
            if paths:
                old_path = os.environ.get("LD_LIBRARY_PATH", "")
                new_path = ":".join(paths)
                if old_path:
                    new_path = f"{new_path}:{old_path}"
                os.environ["LD_LIBRARY_PATH"] = new_path
                return True
    return False

if not os.environ.get("NVIDIA_PATHS_SET"):
    if setup_nvidia_paths():
        os.environ["NVIDIA_PATHS_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import tensorflow as tf
print("Devices:", tf.config.list_physical_devices('GPU'))
