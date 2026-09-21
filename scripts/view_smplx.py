import open3d as o3d
from pathlib import Path


PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
OBJ_PATH = PROJECT_ROOT / "outputs" / "smplx_neutral.obj"


print("Loading:", OBJ_PATH)

mesh = o3d.io.read_triangle_mesh(str(OBJ_PATH))

mesh.compute_vertex_normals()

print("Vertices:", len(mesh.vertices))
print("Triangles:", len(mesh.triangles))

o3d.visualization.draw_geometries(
    [mesh],
    window_name="SMPL-X Neutral Human",
    width=1000,
    height=800
)
