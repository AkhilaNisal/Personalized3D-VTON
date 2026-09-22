# MediaPipe landmark index -> SMPL-X joint index

MEDIAPIPE_TO_SMPLX = {
    "left_shoulder": {
        "mediapipe": 11,
        "smplx": 16,
    },

    "right_shoulder": {
        "mediapipe": 12,
        "smplx": 17,
    },

    "left_elbow": {
        "mediapipe": 13,
        "smplx": 18,
    },

    "right_elbow": {
        "mediapipe": 14,
        "smplx": 19,
    },

    "left_wrist": {
        "mediapipe": 15,
        "smplx": 20,
    },

    "right_wrist": {
        "mediapipe": 16,
        "smplx": 21,
    },

    "left_hip": {
        "mediapipe": 23,
        "smplx": 1,
    },

    "right_hip": {
        "mediapipe": 24,
        "smplx": 2,
    },

    "left_knee": {
        "mediapipe": 25,
        "smplx": 4,
    },

    "right_knee": {
        "mediapipe": 26,
        "smplx": 5,
    },

    "left_ankle": {
        "mediapipe": 27,
        "smplx": 7,
    },

    "right_ankle": {
        "mediapipe": 28,
        "smplx": 8,
    },
}


# Convenient lists for fitting

BODY_JOINT_NAMES = list(MEDIAPIPE_TO_SMPLX.keys())

MEDIAPIPE_INDICES = [
    MEDIAPIPE_TO_SMPLX[name]["mediapipe"]
    for name in BODY_JOINT_NAMES
]

SMPLX_INDICES = [
    MEDIAPIPE_TO_SMPLX[name]["smplx"]
    for name in BODY_JOINT_NAMES
]