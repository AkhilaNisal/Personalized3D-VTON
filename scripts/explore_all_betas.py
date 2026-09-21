import torch
import smplx
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "all_betas"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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

for beta_index in range(10):

    print(f"\nGenerating beta_{beta_index + 1}...")

    betas = torch.zeros(1, 10, device=device)

    # Change only one beta parameter
    betas[0, beta_index] = 3.0

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

    output_file = OUTPUT_PATH / f"beta_{beta_index + 1}_plus3.obj"

    mesh.export(output_file)

    print("Saved:", output_file)

print("\nAll beta experiments completed!")