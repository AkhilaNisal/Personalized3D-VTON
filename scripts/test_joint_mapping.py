from joint_mapping import (
    MEDIAPIPE_TO_SMPLX,
    BODY_JOINT_NAMES,
    MEDIAPIPE_INDICES,
    SMPLX_INDICES,
)


print("=" * 70)
print("MEDIAPIPE → SMPL-X JOINT MAPPING")
print("=" * 70)

for name in BODY_JOINT_NAMES:

    mapping = MEDIAPIPE_TO_SMPLX[name]

    print(
        f"{name:16s} "
        f"MediaPipe={mapping['mediapipe']:2d}  "
        f"SMPL-X={mapping['smplx']:2d}"
    )


print()
print("MediaPipe indices:", MEDIAPIPE_INDICES)
print("SMPL-X indices:   ", SMPLX_INDICES)

print()
print("Mapping contains", len(BODY_JOINT_NAMES), "body landmarks.")