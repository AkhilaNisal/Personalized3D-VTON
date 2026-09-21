import torch
import smplx
import trimesh
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shape_experiment"

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

print("\nLoading SMPL-X...")

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

print("SMPL-X loaded!")


# --------------------------------------------------
# Shape values to test
# --------------------------------------------------

beta_values = [-3.0, -1.5, 0.0, 1.5, 3.0]


# --------------------------------------------------
# Generate different body shapes
# --------------------------------------------------

for value in beta_values:

    print(f"\nGenerating beta_1 = {value}")

    betas = torch.zeros(1, 10, device=device)

    # Change only the first shape parameter
    betas[0, 0] = value

    with torch.no_grad():

        output = model(
            betas=betas,
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=torch.zeros(1, 63, device=device),
            left_hand_pose=torch.zeros(1, 45, device=device),
            right_hand_pose=torch.zeros(1, 45, device=device),
            jaw_pose=torch.zeros(1, 3, device=device),
            leye_pose=torch.zeros(1, 3, device=device),
            reye_pose=torch.zeros(1, 3, device=device),
            expression=torch.zeros(1, 10, device=device),
        )

    vertices = output.vertices[0].cpu().numpy()

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=model.faces,
        process=False
    )

    filename = f"beta_1_{value:+.1f}.obj"

    output_file = OUTPUT_PATH / filename

    mesh.export(output_file)

    print("Saved:", output_file)


print("\nShape experiment completed!")