from pathlib import Path

import torch
import smplx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# Load SMPL-X
# ---------------------------------------------------------

model = smplx.create(
    model_path=str(MODEL_PATH),
    model_type="smplx",
    gender="neutral",
    ext="npz",
    num_betas=10,
    use_pca=False,
    batch_size=1,
).to(device)


# ---------------------------------------------------------
# Generate default body
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

print()
print("=" * 70)
print("SMPL-X JOINT INFORMATION")
print("=" * 70)

print("Number of joints:", output.joints.shape[1])
print("Joint tensor shape:", output.joints.shape)


# ---------------------------------------------------------
# Inspect model attributes related to joints
# ---------------------------------------------------------

print()
print("=" * 70)
print("JOINT-RELATED MODEL ATTRIBUTES")
print("=" * 70)

for attribute in [
    "J_regressor",
    "joint_mapper",
    "NUM_BODY_JOINTS",
    "NUM_HAND_JOINTS",
    "NUM_FACE_JOINTS",
    "NUM_JOINTS",
]:

    if hasattr(model, attribute):

        value = getattr(model, attribute)

        if value is None:
            print(f"{attribute}: None")

        elif hasattr(value, "shape"):
            print(f"{attribute}: shape={value.shape}")

        else:
            print(f"{attribute}: {value}")


# ---------------------------------------------------------
# Inspect joint tensor
# ---------------------------------------------------------

print()
print("=" * 70)
print("FIRST 55 SMPL-X JOINTS")
print("=" * 70)

joints = output.joints[0].detach().cpu()

for index in range(min(55, len(joints))):

    x, y, z = joints[index]

    print(
        f"{index:3d}: "
        f"x={x:8.4f}  "
        f"y={y:8.4f}  "
        f"z={z:8.4f}"
    )


# ---------------------------------------------------------
# Inspect body pose structure
# ---------------------------------------------------------

print()
print("=" * 70)
print("BODY POSE")
print("=" * 70)

print("Body pose parameters:", model.NUM_BODY_JOINTS)
print("Each joint uses 3 rotation parameters.")
print("Body pose tensor shape:", (1, model.NUM_BODY_JOINTS * 3))


print()
print("Inspection completed successfully.")