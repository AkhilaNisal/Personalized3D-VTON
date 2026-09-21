\# Personalized3D-VTON



Personalized 3D human reconstruction and real-time virtual try-on system.



\## Project Overview



This project aims to develop a personalized 3D human model from multiple RGB images or video frames and use the reconstructed body for realistic virtual clothing try-on.



The system will investigate:



\- Multi-view human reconstruction

\- Personalized 3D body shape estimation

\- Anthropometric measurement extraction

\- Skin and surface appearance reconstruction

\- 3D garment fitting

\- Cloth deformation and simulation

\- Real-time virtual try-on



\## Planned Pipeline



RGB Video / Multi-view Images

&#x20;       ↓

Person Segmentation

&#x20;       ↓

Human Pose Estimation

&#x20;       ↓

SMPL-X Reconstruction

&#x20;       ↓

Multi-view Shape Optimization

&#x20;       ↓

Personalized 3D Human

&#x20;       ↓

# Personalized3D-VTON

Research project: **Personalized 3D Human Avatar for Real-Time Virtual Clothing Try-On**.

## Overview

The goal is to construct a high-fidelity personalized 3D human avatar from RGB imagery and use it as the body representation for realistic virtual clothing try-on. The wording *high-fidelity personalized avatar* is intentional: RGB-based reconstruction cannot guarantee an exact physical digital twin.

## Research Motivation

Image-based virtual try-on can struggle with body-specific geometry, clothing fit, loose garments, pose changes, temporal consistency, and physically plausible 3D garment behavior. This project investigates a body-centered pipeline in which the person's shape and pose are represented explicitly before garments are fitted and deformed.

## Main Research Idea

Multi-view RGB capture -> personalized 3D body -> garment representation -> pose-aware clothing deformation -> virtual try-on.

## System Architecture

```text
Multi-view RGB camera/video
	|
Person segmentation and preprocessing
	|
Pose estimation and multi-view alignment
	|
3D human body reconstruction
	|
SMPL-X shape parameters beta
	|
Personalized 3D human avatar
	|
Anthropometric measurements
	|
3D garment representation
	|
Cloth fitting, deformation, and simulation
	|
Pose-aware virtual try-on
	|
Real-time mirror-like output
```

Realistic clothing behavior such as body-specific fit, stretching, hanging, folds, pose dependence, and temporal consistency is a long-term objective. These features are not implemented yet.

## Current Development Status

| Component | Status |
| --- | --- |
| SMPL-X environment | Completed |
| SMPL-X model loading | Completed |
| GPU acceleration | Completed |
| Neutral mesh generation | Completed |
| SMPL-X visualization | Completed |
| beta shape experiments | Completed |
| 10-beta experiment | Completed |
| theta pose experiment | Completed |
| Single-image body estimation | Planned |
| Multi-view body reconstruction | Planned |
| Anthropometric validation | Planned |
| Garment reconstruction | Planned |
| Cloth simulation | Planned |
| Real-time VTON | Planned |

### Completed Environment

- Windows 11
- Miniconda environment `human3d`
- Python 3.11
- NVIDIA GeForce RTX 4050 Laptop GPU, approximately 6 GB VRAM
- PyTorch 2.11.0+cu128 with CUDA available through PyTorch
- Verified libraries: OpenCV, NumPy, Open3D, trimesh, and smplx

### Completed SMPL-X Work

The official SMPL-X model was downloaded separately and placed locally. `scripts/test_smplx.py` loads the neutral model with CUDA, generates a default neutral human, and exports `outputs/smplx_neutral.obj`. `scripts/view_smplx.py` opens that mesh with Open3D.

The shape experiment varies only beta 1 over `-3.0`, `-1.5`, `0.0`, `+1.5`, and `+3.0`, with results in `outputs/shape_experiment/`. The all-beta experiment independently sets beta 1 through beta 10 to `+3.0`, with results in `outputs/all_betas/`. These betas are learned shape directions and should not be interpreted as direct physical labels such as height, chest, or fatness.

The pose experiment changes body pose parameters while keeping shape fixed. It includes neutral standing, left-arm variation, and right-arm variation in `outputs/pose_experiment/`. Together, these experiments demonstrate the distinction between beta (body shape) and theta (body pose).

## Project Structure

```text
Personalized3D-VTON/
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   ├── frames/
│   └── processed/
├── models/
│   ├── README.md
│   └── smplx/
│       └── official/
│           └── smplx/
├── scripts/
│   ├── test.py
│   ├── test_smplx.py
│   ├── view_smplx.py
│   ├── explore_shape.py
│   ├── view_shape_experiment.py
│   ├── explore_all_betas.py
│   ├── view_all_betas.py
│   ├── explore_pose.py
│   └── view_pose_experiment.py
├── notebooks/
├── outputs/
│   ├── shape_experiment/
│   ├── all_betas/
│   └── pose_experiment/
└── requirements/
    └── environment.txt
```

Model weights, private data, and generated outputs are intentionally ignored by Git. See `models/README.md` for model setup.

## Installation

The currently working environment is documented in [requirements/environment.txt](requirements/environment.txt). The commands below create a compatible Conda environment; do not run them inside an existing environment unless you intend to recreate it.

```powershell
conda create -n human3d python=3.11 -y
conda activate human3d

# Install the PyTorch CUDA build using the command for your platform from:
# https://pytorch.org/get-started/locally/
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install opencv-python numpy open3d trimesh smplx
```

Download the official SMPL-X model files according to the license and download procedure, then place them in `models/smplx/official/smplx/`. The required files are `SMPLX_NEUTRAL.npz`, `SMPLX_MALE.npz`, and `SMPLX_FEMALE.npz`. A separate CUDA Toolkit installation is not required when the installed PyTorch package provides the CUDA runtime needed by these scripts.

## Verify GPU

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Running the Experiments

Run these commands from the repository root. The visualization commands open Open3D windows.

```powershell
python scripts\test.py
python scripts\test_smplx.py
python scripts\view_smplx.py
python scripts\explore_shape.py
python scripts\view_shape_experiment.py
python scripts\explore_all_betas.py
python scripts\view_all_betas.py
python scripts\explore_pose.py
python scripts\view_pose_experiment.py
```

## Understanding SMPL-X

SMPL-X is a parametric 3D human model. Its output can be written conceptually as:

```text
M(beta, theta, psi)
```

- `beta`: body shape parameters, represented as learned shape directions
- `theta`: body pose parameters
- `psi`: facial expression parameters

Personalization will ultimately come from estimating these parameters from observations of a particular person. The current work controls them directly; it does not yet reconstruct a real person from RGB imagery.

## Research Roadmap

1. **Stage 1 - SMPL-X fundamentals:** Completed
2. **Stage 2 - Single-image 3D human estimation:** Planned / next
3. **Stage 3 - Anthropometric measurement extraction:** Planned
4. **Stage 4 - Multi-view RGB capture and body-shape optimization:** Planned
5. **Stage 5 - High-fidelity personalized avatar:** Planned
6. **Stage 6 - 3D garment representation:** Planned
7. **Stage 7 - Garment-body interaction and cloth deformation:** Planned
8. **Stage 8 - Pose-aware virtual try-on:** Planned
9. **Stage 9 - Temporal consistency:** Planned
10. **Stage 10 - Real-time optimization:** Planned

## Research Questions

1. How accurately can SMPL-X body shape parameters be estimated from RGB imagery?
2. How much does multi-view capture improve personalized reconstruction compared with a single image?
3. How accurately can anthropometric measurements be extracted from the reconstructed body?
4. How can a personalized body representation improve virtual clothing fit?
5. How can garment deformation be modeled for different body shapes and poses?
6. How can the system maintain temporal consistency for real-time video?

## Limitations

- RGB images do not guarantee exact physical body geometry.
- SMPL-X is a learned parametric representation.
- Initial experiments use a neutral body model.
- Current experiments are parameter-control experiments, not real-person reconstruction.
- Garment simulation and real-time performance are not implemented yet.

## References

- [SMPL-X project](https://smpl-x.is.tue.mpg.de/)
- [Expressive Body Capture: 3D Hands, Face, and Body from a Single Image](https://arxiv.org/abs/1904.05866)
- [4DHumans / HMR2](https://github.com/shubham-goel/4D-Humans)
- [PARE](https://github.com/mkocabas/PARE)
- [DensePose](https://github.com/facebookresearch/DensePose)
- [ICON](https://github.com/YuliangXiu/ICON)
- [ECON](https://github.com/YuliangXiu/ECON)
- [VITON-HD](https://github.com/shadowpa0327/VITON-HD)
- [StableVITON](https://github.com/rlawjdghek/StableVITON)
- [OOTDiffusion](https://github.com/levihsu/OOTDiffusion)
- [CatVTON](https://github.com/Zheng-Chong/CatVTON)
- [DPIDM](https://github.com/tencent-ailab/3dgrids)

## Author

**Akhila Nisal Wedamestrige**
University of Moratuwa
Department of Electronic and Telecommunication Engineering

- GitHub: [https://github.com/AkhilaNisal](https://github.com/AkhilaNisal)
- Portfolio: [https://akhilanisal.github.io/akhila-portfolio/](https://akhilanisal.github.io/akhila-portfolio/)

