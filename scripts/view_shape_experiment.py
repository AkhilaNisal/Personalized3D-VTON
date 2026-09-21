import open3d as o3d
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
SHAPE_PATH = PROJECT_ROOT / "outputs" / "shape_experiment"

values = [-3.0, -1.5, 0.0, 1.5, 3.0]

meshes = []

for value in values:

    filename = f"beta_1_{value:+.1f}.obj"
    path = SHAPE_PATH / filename

    print("Loading:", path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()

    # Move each model horizontally so they don't overlap
    mesh.translate((values.index(value) * 1.2, 0, 0))

    meshes.append(mesh)

print("\nLoaded", len(meshes), "models.")

o3d.visualization.draw_geometries(
    meshes,
    window_name="SMPL-X Beta 1 Shape Experiment",
    width=1400,
    height=800
)