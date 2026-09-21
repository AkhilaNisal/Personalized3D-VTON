import torch
import cv2
import numpy as np
import open3d as o3d
import trimesh

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Open3D:", o3d.__version__)
print("Trimesh:", trimesh.__version__)

print("\nEnvironment OK!")