import kagglehub
import os
import shutil

zip_path = kagglehub.dataset_download("akhatova/pcb-defects", force_download=True)

cwd = os.getcwd()
download_dir = os.path.join(cwd, "Data")
os.makedirs(download_dir, exist_ok=True)

if os.path.exists(zip_path):
    dst_path = os.path.join(download_dir, os.path.basename(zip_path))
    shutil.move(zip_path, dst_path)
    print("Moved dataset to:", dst_path)
else:
    print("Download failed — file not found:", zip_path)
