import cv2
import numpy as np
import torch
import smplx
import mediapipe as mp

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(r"D:\Personalized3D-VTON")

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "smplx"
    / "official"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "person.jpg"
)

POSE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "pose_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "smplx_pose_fitted.jpg"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load image
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

height, width = image.shape[:2]

print(
    f"Image size: {width} x {height}"
)


# ============================================================
# Run MediaPipe Pose
# ============================================================

print("Running MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=str(
        POSE_MODEL_PATH
    )
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)

with vision.PoseLandmarker.create_from_options(
    options
) as landmarker:

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    result = landmarker.detect(
        mp_image
    )


if not result.pose_landmarks:

    raise RuntimeError(
        "MediaPipe pose was not detected."
    )

landmarks = result.pose_landmarks[0]

print(
    "MediaPipe pose detected."
)

print(
    "Landmarks:",
    len(landmarks)
)


# ============================================================
# MediaPipe landmark indices
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
# Extract MediaPipe 2D target points
# ============================================================

target_2d = []

confidence = []

for name, index in mp_indices.items():

    landmark = landmarks[index]

    x = landmark.x * width
    y = landmark.y * height

    target_2d.append(
        [x, y]
    )

    confidence.append(
        landmark.visibility
    )


target_2d = torch.tensor(
    target_2d,
    dtype=torch.float32,
    device=device,
)

confidence = torch.tensor(
    confidence,
    dtype=torch.float32,
    device=device,
)


print()
print("Target joints:")

for name, point, conf in zip(
    mp_indices.keys(),
    target_2d.cpu().numpy(),
    confidence.cpu().numpy(),
):

    print(
        f"{name:16s}"
        f" x={point[0]:7.2f}"
        f" y={point[1]:7.2f}"
        f" visibility={conf:.3f}"
    )


# ============================================================
# Load SMPL-X
# ============================================================

print()
print("Loading SMPL-X...")

model = smplx.create(
    model_path=str(
        MODEL_PATH
    ),
    model_type="smplx",
    gender="neutral",
    ext="npz",
    num_betas=10,
    use_pca=False,
    batch_size=1,
).to(device)

model.eval()

print("SMPL-X loaded.")


# ============================================================
# Fixed body shape
#
# IMPORTANT:
#
# We are NOT optimizing body shape yet.
#
# β = 0 means the default neutral SMPL-X body.
# ============================================================

betas = torch.zeros(
    1,
    10,
    dtype=torch.float32,
    device=device,
)


# ============================================================
# Learnable SMPL-X body pose
#
# 21 body joints × 3 axis-angle parameters
#
# 21 × 3 = 63
# ============================================================

body_pose = torch.zeros(
    1,
    63,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)


# ============================================================
# Learnable global orientation
#
# This rotates the entire body.
# ============================================================

global_orient = torch.zeros(
    1,
    3,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)


# ============================================================
# Learnable camera scale
#
# We use log(scale) so that scale always remains positive.
# ============================================================

log_scale = torch.tensor(
    np.log(250.0),
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)


# ============================================================
# Learnable camera translation
#
# translation[0] = image X
# translation[1] = image Y
# ============================================================

translation = torch.zeros(
    1,
    2,
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)


# ============================================================
# SMPL-X joints corresponding to MediaPipe
# ============================================================

smplx_indices = torch.tensor(
    [
        16, 17,       # shoulders
        18, 19,       # elbows
        20, 21,       # wrists
        1, 2,         # hips
        4, 5,         # knees
        7, 8,         # ankles
    ],
    dtype=torch.long,
    device=device,
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    [
        body_pose,
        global_orient,
        log_scale,
        translation,
    ],
    lr=0.02,
)


# ============================================================
# Optimization settings
# ============================================================

iterations = 1000


print()
print("=" * 70)
print("STARTING SMPL-X POSE FITTING")
print("=" * 70)

print()
print("Fixed:")
print("  β (body shape) = 0")

print()
print("Optimizing:")
print("  θ (body pose)")
print("  global orientation")
print("  camera scale")
print("  camera translation")

print()
print("Iterations:", iterations)

print()


# ============================================================
# Optimization loop
# ============================================================

for iteration in range(iterations):

    optimizer.zero_grad()

    # --------------------------------------------------------
    # Generate SMPL-X body
    # --------------------------------------------------------

    output = model(

        betas=betas,

        body_pose=body_pose,

        global_orient=global_orient,

        transl=torch.zeros(
            1,
            3,
            dtype=torch.float32,
            device=device,
        ),
    )


    # --------------------------------------------------------
    # Get the first 55 SMPL-X regressed joints
    # --------------------------------------------------------

    joints_3d = (
        output.joints[:, :55, :]
    )


    # --------------------------------------------------------
    # Select corresponding joints
    # --------------------------------------------------------

    joints_3d = (
        joints_3d[:, smplx_indices, :]
    )


    # --------------------------------------------------------
    # Weak-perspective projection
    #
    # u = sX + tx
    # v = -sY + ty
    #
    # We use -Y because:
    #
    # SMPL-X:
    #   Y increases upward
    #
    # Image:
    #   V increases downward
    # --------------------------------------------------------

    scale = torch.exp(
        log_scale
    )

    x = joints_3d[:, :, 0]

    y = joints_3d[:, :, 1]


    projected_x = (
        scale * x
        + translation[:, 0:1]
    )

    projected_y = (
        -scale * y
        + translation[:, 1:2]
    )


    projected = torch.stack(
        [
            projected_x,
            projected_y,
        ],
        dim=-1,
    )


    projected = projected[0]


    # ========================================================
    # Joint reprojection loss
    # ========================================================

    difference = (
        projected
        - target_2d
    )


    squared_distance = (
        difference ** 2
    ).sum(
        dim=1
    )


    # Visibility-weighted loss

    joint_loss = (
        squared_distance
        * confidence
    ).mean()


    # ========================================================
    # Pose regularization
    #
    # Prevents the optimizer from producing unnecessarily
    # large rotations.
    # ========================================================

    pose_regularization = (
        body_pose ** 2
    ).mean()


    # ========================================================
    # Total loss
    # ========================================================

    loss = (
        joint_loss
        + 10.0 * pose_regularization
    )


    # ========================================================
    # Backpropagation
    # ========================================================

    loss.backward()

    optimizer.step()


    # ========================================================
    # Progress
    # ========================================================

    if (
        iteration % 100 == 0
        or iteration == iterations - 1
    ):

        rmse = torch.sqrt(
            squared_distance.mean()
        )

        print(
            f"Iteration {iteration:4d} | "
            f"Loss {loss.item():12.4f} | "
            f"RMSE {rmse.item():8.3f} px"
        )


# ============================================================
# Generate final fitted body
# ============================================================

print()
print("Generating final fitted body...")


with torch.no_grad():

    output = model(

        betas=betas,

        body_pose=body_pose,

        global_orient=global_orient,

        transl=torch.zeros(
            1,
            3,
            dtype=torch.float32,
            device=device,
        ),
    )


    joints_3d = (
        output.joints[:, :55, :]
    )


    joints_3d = (
        joints_3d[:, smplx_indices, :]
    )


    # --------------------------------------------------------
    # Final projection
    # --------------------------------------------------------

    scale = torch.exp(
        log_scale
    )


    x = joints_3d[:, :, 0]

    y = joints_3d[:, :, 1]


    projected_x = (
        scale * x
        + translation[:, 0:1]
    )


    projected_y = (
        -scale * y
        + translation[:, 1:2]
    )


    projected = torch.stack(
        [
            projected_x,
            projected_y,
        ],
        dim=-1,
    )


    projected = (
        projected[0]
        .cpu()
        .numpy()
    )


# ============================================================
# Calculate final RMSE
# ============================================================

target_numpy = (
    target_2d
    .cpu()
    .numpy()
)


errors = (
    projected
    - target_numpy
)


distances = np.sqrt(
    np.sum(
        errors ** 2,
        axis=1,
    )
)


final_rmse = np.sqrt(
    np.mean(
        distances ** 2
    )
)


print()
print(
    f"Final joint RMSE: "
    f"{final_rmse:.3f} pixels"
)


# ============================================================
# Draw result
# ============================================================

output_image = image.copy()


# ============================================================
# Skeleton connections
#
# Order:
#
# 0 left shoulder
# 1 right shoulder
# 2 left elbow
# 3 right elbow
# 4 left wrist
# 5 right wrist
# 6 left hip
# 7 right hip
# 8 left knee
# 9 right knee
# 10 left ankle
# 11 right ankle
# ============================================================

connections = [

    # Left arm
    (0, 2),
    (2, 4),

    # Right arm
    (1, 3),
    (3, 5),

    # Torso
    (0, 6),
    (1, 7),

    # Left leg
    (6, 8),
    (8, 10),

    # Right leg
    (7, 9),
    (9, 11),

    # Shoulders
    (0, 1),

    # Hips
    (6, 7),
]


# ============================================================
# Draw MediaPipe target skeleton
#
# Blue
# ============================================================

mp_pixels = target_numpy


for a, b in connections:

    p1 = tuple(
        mp_pixels[a].astype(int)
    )

    p2 = tuple(
        mp_pixels[b].astype(int)
    )

    cv2.line(
        output_image,
        p1,
        p2,
        (255, 0, 0),
        2,
    )


for point in mp_pixels:

    cv2.circle(
        output_image,
        tuple(
            point.astype(int)
        ),
        5,
        (255, 0, 0),
        -1,
    )


# ============================================================
# Draw fitted SMPL-X
#
# Yellow
# ============================================================

for a, b in connections:

    p1 = tuple(
        projected[a].astype(int)
    )

    p2 = tuple(
        projected[b].astype(int)
    )

    cv2.line(
        output_image,
        p1,
        p2,
        (0, 255, 255),
        2,
    )


for point in projected:

    cv2.circle(
        output_image,
        tuple(
            point.astype(int)
        ),
        5,
        (0, 255, 255),
        -1,
    )


# ============================================================
# Legend
# ============================================================

cv2.rectangle(
    output_image,
    (10, 10),
    (255, 80),
    (0, 0, 0),
    -1,
)


# Blue legend

cv2.circle(
    output_image,
    (30, 30),
    6,
    (255, 0, 0),
    -1,
)

cv2.putText(
    output_image,
    "MediaPipe target",
    (45, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1,
    cv2.LINE_AA,
)


# Yellow legend

cv2.circle(
    output_image,
    (30, 58),
    6,
    (0, 255, 255),
    -1,
)

cv2.putText(
    output_image,
    "Fitted SMPL-X",
    (45, 63),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1,
    cv2.LINE_AA,
)


# ============================================================
# Save output
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

cv2.imwrite(
    str(OUTPUT_PATH),
    output_image,
)


# ============================================================
# Print final information
# ============================================================

print()
print("=" * 70)
print("POSE FITTING COMPLETE")
print("=" * 70)

print()
print("Final RMSE:")
print(
    f"{final_rmse:.3f} pixels"
)

print()
print("Output saved to:")
print(OUTPUT_PATH)

print()
print("Blue   = MediaPipe target")
print("Yellow = fitted SMPL-X")

print()
print("Pose fitting experiment completed!")