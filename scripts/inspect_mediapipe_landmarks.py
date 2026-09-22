from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "person.jpg"
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker.task"


# MediaPipe landmark names
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")


# MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
)


rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image,
)


# Detect pose
with PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)


if len(result.pose_landmarks) == 0:
    raise RuntimeError("No pose detected.")


landmarks = result.pose_landmarks[0]


print()
print("=" * 70)
print("MEDIAPIPE POSE LANDMARKS")
print("=" * 70)

print(f"Number of landmarks: {len(landmarks)}")
print()


for index, landmark in enumerate(landmarks):

    name = LANDMARK_NAMES[index]

    print(
        f"{index:2d}  "
        f"{name:20s} "
        f"x={landmark.x:8.4f}  "
        f"y={landmark.y:8.4f}  "
        f"z={landmark.z:8.4f}  "
        f"visibility={landmark.visibility:6.3f}"
    )


print()
print("=" * 70)
print("IMPORTANT BODY LANDMARKS")
print("=" * 70)

important = [
    11, 12,
    13, 14,
    15, 16,
    23, 24,
    25, 26,
    27, 28,
]

for index in important:

    landmark = landmarks[index]

    print(
        f"{LANDMARK_NAMES[index]:15s} "
        f"x={landmark.x:.4f} "
        f"y={landmark.y:.4f} "
        f"z={landmark.z:.4f} "
        f"visibility={landmark.visibility:.3f}"
    )