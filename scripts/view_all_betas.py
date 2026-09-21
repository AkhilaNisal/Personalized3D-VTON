import open3d as o3d
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
BETA_PATH = PROJECT_ROOT / "outputs" / "all_betas"

meshes = []

for beta_index in range(10):

    filename = f"beta_{beta_index + 1}_plus3.obj"
    path = BETA_PATH / filename

    print("Loading:", path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh.compute_vertex_normals()

    # Arrange the models in two rows
    row = beta_index // 5
    column = beta_index % 5

    mesh.translate((
        column * 1.2,
        0,
        row * -2.2
    ))

    meshes.append(mesh)

print("\nLoaded", len(meshes), "models.")

o3d.visualization.draw_geometries(
    meshes,
    window_name="SMPL-X — All 10 Shape Parameters",
    width=1400,
    height=900
)