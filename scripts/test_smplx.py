import torch
import smplx
import trimesh
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")

MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"

OUTPUT_PATH = PROJECT_ROOT / "outputs"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load SMPL-X
# --------------------------------------------------

print("\nLoading SMPL-X model...")

model = smplx.create(
    model_path=str(MODEL_PATH),
    model_type="smplx",
    gender="neutral",
    ext="npz",
    num_betas=10,
    use_pca=False,
    batch_size=1,
)

model = model.to(device)

print("SMPL-X model loaded!")


# --------------------------------------------------
# Generate default human
# --------------------------------------------------

print("\nGenerating human mesh...")

with torch.no_grad():

    output = model(
        betas=torch.zeros(1, 10, device=device),
        global_orient=torch.zeros(1, 3, device=device),
        body_pose=torch.zeros(1, 63, device=device),
        left_hand_pose=torch.zeros(1, 45, device=device),
        right_hand_pose=torch.zeros(1, 45, device=device),
        jaw_pose=torch.zeros(1, 3, device=device),
        leye_pose=torch.zeros(1, 3, device=device),
        reye_pose=torch.zeros(1, 3, device=device),
        expression=torch.zeros(1, 10, device=device),
    )


# --------------------------------------------------
# Convert to NumPy
# --------------------------------------------------

vertices = output.vertices[0].detach().cpu().numpy()

faces = model.faces


print("Vertices:", vertices.shape)
print("Faces:", faces.shape)


# --------------------------------------------------
# Create mesh
# --------------------------------------------------

mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False
)


# --------------------------------------------------
# Save OBJ
# --------------------------------------------------

obj_path = OUTPUT_PATH / "smplx_neutral.obj"

mesh.export(obj_path)

print("\nMesh saved to:")
print(obj_path)

print("\nSMPL-X test completed successfully!")