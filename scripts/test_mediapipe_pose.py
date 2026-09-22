from pathlib import Path

import cv2
import mediapipe as mp


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "person.jpg"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "mediapipe_pose.jpg"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

print("Image loaded:", IMAGE_PATH)
print("Image size:", image.shape)


# ---------------------------------------------------------
# MediaPipe Pose Landmarker
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# The pose model will be added here.
# We will download the model separately in the next step.

MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Pose model not found:\n{MODEL_PATH}\n"
        "Download the MediaPipe Pose Landmarker model first."
    )


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ---------------------------------------------------------
# Run pose estimation
# ---------------------------------------------------------

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image,
)


with PoseLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)


# ---------------------------------------------------------
# Check result
# ---------------------------------------------------------

print()
print("Number of detected poses:", len(result.pose_landmarks))

if len(result.pose_landmarks) == 0:
    print("No person detected.")
    raise SystemExit

landmarks = result.pose_landmarks[0]

print("Number of landmarks:", len(landmarks))

print()
print("First few landmarks:")

for i, landmark in enumerate(landmarks[:10]):
    print(
        f"{i:2d}: "
        f"x={landmark.x:.4f}, "
        f"y={landmark.y:.4f}, "
        f"z={landmark.z:.4f}"
    )


# ---------------------------------------------------------
# Draw landmarks
# ---------------------------------------------------------

annotated = image.copy()

height, width = annotated.shape[:2]

for landmark in landmarks:

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    if 0 <= x < width and 0 <= y < height:
        cv2.circle(
            annotated,
            (x, y),
            5,
            (0, 255, 0),
            -1,
        )


# Draw connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),

    (9, 10),

    (11, 12),

    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),

    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25), (25, 27),
    (27, 29), (27, 31),

    (24, 26), (26, 28),
    (28, 30), (28, 32),
]

for a, b in connections:

    if a >= len(landmarks) or b >= len(landmarks):
        continue

    x1 = int(landmarks[a].x * width)
    y1 = int(landmarks[a].y * height)

    x2 = int(landmarks[b].x * width)
    y2 = int(landmarks[b].y * height)

    cv2.line(
        annotated,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2,
    )


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

cv2.imwrite(str(OUTPUT_PATH), annotated)

print()
print("Pose visualization saved to:")
print(OUTPUT_PATH)