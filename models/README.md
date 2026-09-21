# SMPL-X Models

SMPL-X is a parametric 3D human body model used in this project for body shape and pose experiments.

This repository does not redistribute the official SMPL-X model weights. Download the files from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) and comply with the applicable license and download terms.

Place the downloaded files in this local directory:

```text
models/smplx/official/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_MALE.npz
└── SMPLX_FEMALE.npz
```

The `.npz` files are excluded from Git. The experiment scripts use the local model directory and will not run until the required model files are available.
