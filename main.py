"""
main.py — Entry point for the Solar Power Prediction application.

Run with:
    python main.py
"""
import os
import sys

# Configure LD_LIBRARY_PATH for nvidia pip packages before importing anything else
if not os.environ.get("NVIDIA_PATHS_SET"):
    try:
        import site
        # Getsitepackages works in most standard python environments
        site_packages = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        
        # Virtualenv fallback
        if hasattr(site, "getusersitepackages"):
            site_packages.append(site.getusersitepackages())
            
        import sysconfig
        if sysconfig.get_path("purelib"):
            site_packages.append(sysconfig.get_path("purelib"))
        for site_pkg in site_packages:
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
                    os.environ["NVIDIA_PATHS_SET"] = "1"
                    os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import customtkinter as ctk
from app import SolarPowerApp

if __name__ == "__main__":
    # Set global appearance before creating any widgets
    ctk.set_appearance_mode("dark")

    app = SolarPowerApp()
    app.mainloop()
