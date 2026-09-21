import torch
import smplx
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")
MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "pose_experiment"

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


def generate_pose(name, body_pose):

    print(f"\nGenerating pose: {name}")

    betas = torch.zeros(1, 10, device=device)

    body_pose_tensor = torch.zeros(
        1, 63,
        device=device
    )

    # Replace body pose with our chosen pose
    body_pose_tensor[0] = torch.tensor(
        body_pose,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        output = model(
            betas=betas,
            global_orient=torch.zeros(1, 3, device=device),
            body_pose=body_pose_tensor,
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

    output_file = OUTPUT_PATH / f"{name}.obj"

    mesh.export(output_file)

    print("Saved:", output_file)


# ------------------------------------------------
# Pose 1: Neutral standing
# ------------------------------------------------

neutral_pose = [0.0] * 63

generate_pose(
    "pose_1_neutral",
    neutral_pose
)


# ------------------------------------------------
# Pose 2: Left arm raised
# ------------------------------------------------

left_arm_pose = [0.0] * 63

# Left shoulder joint
left_arm_pose[15 * 3 + 2] = -1.0

generate_pose(
    "pose_2_left_arm",
    left_arm_pose
)


# ------------------------------------------------
# Pose 3: Right arm raised
# ------------------------------------------------

right_arm_pose = [0.0] * 63

# Right shoulder joint
right_arm_pose[16 * 3 + 2] = 1.0

generate_pose(
    "pose_3_right_arm",
    right_arm_pose
)


print("\nPose experiment completed!")