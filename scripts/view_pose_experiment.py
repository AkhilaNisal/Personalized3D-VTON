import open3d as o3d
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
POSE_PATH = PROJECT_ROOT / "outputs" / "pose_experiment"

pose_files = [
    "pose_1_neutral.obj",
    "pose_2_left_arm.obj",
    "pose_3_right_arm.obj",
]

meshes = []

for i, filename in enumerate(pose_files):

    path = POSE_PATH / filename

    print("Loading:", path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()

    # Arrange models horizontally
    mesh.translate((i * 1.5, 0, 0))

    meshes.append(mesh)

print("\nLoaded", len(meshes), "poses.")

o3d.visualization.draw_geometries(
    meshes,
    window_name="SMPL-X Pose Experiment",
    width=1400,
    height=800
)