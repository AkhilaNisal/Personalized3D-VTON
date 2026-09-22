import cv2
import numpy as np
import torch
import smplx
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")

MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"
IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "person.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "smplx_projection.jpg"


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load image
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width = image.shape[:2]

print(f"Image size: {width} x {height}")


# ============================================================
# Load SMPL-X
# ============================================================

print("Loading SMPL-X model...")

model = smplx.create(
    model_path=str(MODEL_PATH),
    model_type="smplx",
    gender="neutral",
    ext="npz",
    num_betas=10,
    use_pca=False,
    batch_size=1,
).to(device)

model.eval()

print("SMPL-X loaded!")


# ============================================================
# Generate neutral SMPL-X body
# ============================================================

print("Generating neutral body...")

with torch.no_grad():

    output = model(
        betas=torch.zeros(1, 10, device=device),
        body_pose=torch.zeros(1, 63, device=device),
        global_orient=torch.zeros(1, 3, device=device),
        transl=torch.zeros(1, 3, device=device),
    )

# First 55 joints correspond to the model's regressed joints.
joints_3d = output.joints[:, :55, :]

joints_3d = joints_3d[0].cpu().numpy()

print("3D joints:", joints_3d.shape)


# ============================================================
# Select the 12 body joints
# ============================================================

smplx_indices = [
    16, 17,       # shoulders
    18, 19,       # elbows
    20, 21,       # wrists
    1, 2,         # hips
    4, 5,         # knees
    7, 8,         # ankles
]

joint_names = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

selected_joints = joints_3d[smplx_indices]


# ============================================================
# Convert SMPL-X coordinates
#
# SMPL-X uses approximately:
#   X = left/right
#   Y = vertical
#   Z = depth
#
# Image coordinates:
#   u = horizontal
#   v = vertical
# ============================================================

x = selected_joints[:, 0]
y = selected_joints[:, 1]


# ============================================================
# Normalize SMPL-X coordinates
#
# We are NOT doing physical camera calibration yet.
#
# This is only a visualization experiment.
# ============================================================

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Add margin around the projected body
margin = 0.15

x_range = x_max - x_min
y_range = y_max - y_min

x_min -= margin * x_range
x_max += margin * x_range

y_min -= margin * y_range
y_max += margin * y_range


# Map SMPL-X X coordinate -> image horizontal coordinate
u = (x - x_min) / (x_max - x_min) * (width - 1)

# Map SMPL-X Y coordinate -> image vertical coordinate.
#
# SMPL-X Y increases upward,
# image V increases downward.
v = (y_max - y) / (y_max - y_min) * (height - 1)

projected = np.column_stack((u, v))


# ============================================================
# Draw projected SMPL-X joints
# ============================================================

output_image = image.copy()


# Skeleton connections
connections = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),

    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
]


name_to_index = {
    name: i
    for i, name in enumerate(joint_names)
}


# Draw skeleton
for name_a, name_b in connections:

    i = name_to_index[name_a]
    j = name_to_index[name_b]

    p1 = tuple(projected[i].astype(int))
    p2 = tuple(projected[j].astype(int))

    cv2.line(
        output_image,
        p1,
        p2,
        (0, 255, 255),
        2,
    )


# Draw joints
for i, name in enumerate(joint_names):

    point = tuple(projected[i].astype(int))

    cv2.circle(
        output_image,
        point,
        6,
        (0, 255, 255),
        -1,
    )

    cv2.putText(
        output_image,
        name,
        (point[0] + 6, point[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Save
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

cv2.imwrite(
    str(OUTPUT_PATH),
    output_image,
)

print()
print("Projected joints:")
print()

for name, point in zip(joint_names, projected):

    print(
        f"{name:16s} "
        f"u={point[0]:7.2f} "
        f"v={point[1]:7.2f}"
    )

print()
print("Output saved to:")
print(OUTPUT_PATH)

print()
print("Projection experiment completed!")