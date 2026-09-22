import cv2
import numpy as np
import torch
import smplx
import mediapipe as mp
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")

MODEL_PATH = PROJECT_ROOT / "models" / "smplx" / "official"
IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "person.jpg"
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker.task"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "compare_mediapipe_smplx.jpg"


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
    raise FileNotFoundError(IMAGE_PATH)

height, width = image.shape[:2]

print(f"Image size: {width} x {height}")


# ============================================================
# MediaPipe Pose
# ============================================================

print("Running MediaPipe Pose...")

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path=str(POSE_MODEL_PATH)
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect(mp_image)


if not result.pose_landmarks:
    raise RuntimeError("MediaPipe could not detect a person.")

mp_landmarks = result.pose_landmarks[0]

print("MediaPipe pose detected.")
print("Landmarks:", len(mp_landmarks))


# ============================================================
# MediaPipe joint indices
# ============================================================

mp_indices = {
    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

    "left_hip": 23,
    "right_hip": 24,

    "left_knee": 25,
    "right_knee": 26,

    "left_ankle": 27,
    "right_ankle": 28,
}


# ============================================================
# Convert MediaPipe coordinates to pixels
# ============================================================

mp_points = {}

for name, index in mp_indices.items():

    landmark = mp_landmarks[index]

    x = landmark.x * width
    y = landmark.y * height

    mp_points[name] = np.array([x, y])


# ============================================================
# Load SMPL-X
# ============================================================

print("Loading SMPL-X...")

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


# ============================================================
# Generate neutral SMPL-X
# ============================================================

with torch.no_grad():

    output = model(
        betas=torch.zeros(1, 10, device=device),
        body_pose=torch.zeros(1, 63, device=device),
        global_orient=torch.zeros(1, 3, device=device),
        transl=torch.zeros(1, 3, device=device),
    )


joints_3d = output.joints[0, :55].cpu().numpy()


# ============================================================
# SMPL-X mapping
# ============================================================

smplx_indices = {
    "left_shoulder": 16,
    "right_shoulder": 17,

    "left_elbow": 18,
    "right_elbow": 19,

    "left_wrist": 20,
    "right_wrist": 21,

    "left_hip": 1,
    "right_hip": 2,

    "left_knee": 4,
    "right_knee": 5,

    "left_ankle": 7,
    "right_ankle": 8,
}


# ============================================================
# Initial weak-perspective normalization
#
# This is NOT fitting yet.
# It only places the neutral SMPL-X skeleton
# into approximately the same image region.
# ============================================================

smplx_points_3d = np.array([
    joints_3d[smplx_indices[name]]
    for name in mp_indices
])

x = smplx_points_3d[:, 0]
y = smplx_points_3d[:, 1]

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

# Match SMPL-X bounding box to MediaPipe bounding box

mp_array = np.array([
    mp_points[name]
    for name in mp_indices
])

mp_x_min, mp_x_max = mp_array[:, 0].min(), mp_array[:, 0].max()
mp_y_min, mp_y_max = mp_array[:, 1].min(), mp_array[:, 1].max()

smplx_width = x_max - x_min
smplx_height = y_max - y_min

mp_width = mp_x_max - mp_x_min
mp_height = mp_y_max - mp_y_min

scale_x = mp_width / smplx_width
scale_y = mp_height / smplx_height

scale = (scale_x + scale_y) / 2.0


# Center alignment

smplx_center_x = (x_min + x_max) / 2
smplx_center_y = (y_min + y_max) / 2

mp_center_x = (mp_x_min + mp_x_max) / 2
mp_center_y = (mp_y_min + mp_y_max) / 2


smplx_projected = []

for point in smplx_points_3d:

    px = (
        (point[0] - smplx_center_x)
        * scale
        + mp_center_x
    )

    py = (
        -(point[1] - smplx_center_y)
        * scale
        + mp_center_y
    )

    smplx_projected.append([px, py])


smplx_projected = np.array(smplx_projected)


# ============================================================
# Skeleton connections
# ============================================================

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
    for i, name in enumerate(mp_indices)
}


# ============================================================
# Draw MediaPipe skeleton
# ============================================================

output = image.copy()

for name_a, name_b in connections:

    p1 = tuple(mp_points[name_a].astype(int))
    p2 = tuple(mp_points[name_b].astype(int))

    cv2.line(
        output,
        p1,
        p2,
        (255, 0, 0),
        2,
    )


for name, point in mp_points.items():

    p = tuple(point.astype(int))

    cv2.circle(
        output,
        p,
        6,
        (255, 0, 0),
        -1,
    )


# ============================================================
# Draw SMPL-X skeleton
# ============================================================

for name_a, name_b in connections:

    i = name_to_index[name_a]
    j = name_to_index[name_b]

    p1 = tuple(smplx_projected[i].astype(int))
    p2 = tuple(smplx_projected[j].astype(int))

    cv2.line(
        output,
        p1,
        p2,
        (0, 255, 255),
        2,
    )


for i, name in enumerate(mp_indices):

    p = tuple(smplx_projected[i].astype(int))

    cv2.circle(
        output,
        p,
        5,
        (0, 255, 255),
        -1,
    )


# ============================================================
# Legend
# ============================================================

cv2.rectangle(
    output,
    (10, 10),
    (250, 75),
    (0, 0, 0),
    -1,
)

cv2.circle(output, (30, 30), 6, (255, 0, 0), -1)
cv2.putText(
    output,
    "MediaPipe - real person",
    (45, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1,
)

cv2.circle(output, (30, 55), 6, (0, 255, 255), -1)
cv2.putText(
    output,
    "SMPL-X - neutral",
    (45, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1,
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
    output,
)

print()
print("Output saved to:")
print(OUTPUT_PATH)

print()
print("Blue   = MediaPipe actual person")
print("Yellow = neutral SMPL-X")

print()
print("Comparison completed!")