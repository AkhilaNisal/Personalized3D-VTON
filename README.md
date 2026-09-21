# Personalized3D-VTON

**Personalized 3D Human Avatar for Real-Time Virtual Clothing Try-On**

Research project by Akhila Nisal Wedamestrige, University of Moratuwa.

## Overview

Personalized3D-VTON investigates how a person's body shape and pose can be represented as a high-fidelity 3D human avatar and used as the body representation for realistic virtual clothing try-on.

The long-term system is intended to use RGB images or video, estimate a person's body geometry and motion, represent the body with a parametric model, and fit garments to that body. The phrase *high-fidelity personalized 3D avatar* is used deliberately. RGB-based reconstruction cannot guarantee an exact physical digital twin, so the project will evaluate reconstruction quality rather than claim perfect physical recovery.

## Research Motivation

Many image-based virtual try-on systems focus on producing a convincing dressed image. They can be less suitable when the goal is a reusable 3D body representation that supports different views, poses, garments, and time steps. Important challenges include:

- body-specific geometry and proportions;
- reliable clothing fit for different body shapes;
- loose, oversized, and tight garments;
- garment deformation when the body moves;
- stretching, hanging, folds, and body-garment contact;
- consistent output across video frames; and
- computational efficiency for a real-time, mirror-like experience.

This project begins with SMPL-X fundamentals and will progressively connect body reconstruction, anthropometric validation, garment representation, cloth deformation, and virtual try-on.

## Main Research Idea

```text
Multi-view RGB capture
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
Garment fitting, deformation, and cloth simulation
        |
Pose-aware virtual try-on
        |
Real-time mirror-like output
```

The complete pipeline shown above is the research target. Only the SMPL-X environment, model loading, mesh generation, visualization, shape controls, and pose controls are currently implemented.

## Current Development Status

| Component | Status |
| --- | --- |
| Windows/Conda/Python development environment | Completed |
| SMPL-X model loading | Completed |
| PyTorch CUDA execution | Completed |
| Neutral SMPL-X mesh generation | Completed |
| SMPL-X mesh visualization | Completed |
| Single beta shape experiment | Completed |
| Independent beta 1-10 experiment | Completed |
| Body pose theta experiment | Completed |
| Single-image body estimation | Planned / next |
| Multi-view body reconstruction | Planned |
| Anthropometric validation | Planned |
| Garment reconstruction | Planned |
| Cloth deformation and simulation | Planned |
| Pose-aware virtual try-on | Planned |
| Temporal consistency | Planned |
| Real-time optimization | Planned |

### Verified Environment

The current working environment is:

- Operating system: Windows 11
- Conda distribution: Miniconda
- Conda environment: `human3d`
- Python: 3.11
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, approximately 6 GB VRAM
- PyTorch: `2.11.0+cu128`
- CUDA available through PyTorch: `True`
- OpenCV: `5.0.0`
- NumPy: `2.4.6`
- Open3D: `0.20.0`
- trimesh: `5.1.0`
- smplx: installed and verified by the SMPL-X scripts

The environment smoke test in `scripts/test.py` prints these dependency versions and confirms CUDA availability. The environment record is maintained in [requirements/environment.txt](requirements/environment.txt).

## Completed Experiments

### 1. Neutral SMPL-X Mesh

`scripts/test_smplx.py`:

- loads the neutral SMPL-X model;
- selects CUDA when available;
- creates a default neutral human with ten body shape parameters;
- exports an OBJ mesh; and
- reports the generated vertex and face dimensions.

The generated mesh is `outputs/smplx_neutral.obj`. It was opened successfully with `scripts/view_smplx.py`, which uses Open3D for visualization.

### 2. Single Shape Direction Experiment

`scripts/explore_shape.py` varies only the first shape coefficient, beta 1, using:

```text
-3.0, -1.5, 0.0, +1.5, +3.0
```

The remaining shape coefficients, pose, hand pose, jaw pose, eye pose, and expression are held at zero. The resulting meshes are written to `outputs/shape_experiment/` and can be viewed together with `scripts/view_shape_experiment.py`.

Beta directions are learned directions in the model's shape space. Beta 1 must not be described as directly meaning a physical property such as height, chest size, or fatness.

### 3. Independent Ten-Beta Experiment

`scripts/explore_all_betas.py` changes beta 1 through beta 10 independently. For each generated mesh, one beta is set to `+3.0` and the other nine are zero. The results are written to `outputs/all_betas/` and viewed with `scripts/view_all_betas.py`.

This experiment is intended to build intuition about the independent learned shape directions. It is not a learned body measurement or a real-person reconstruction experiment.

### 4. Body Pose Experiment

`scripts/explore_pose.py` keeps the shape coefficients at zero and changes body pose parameters. It generates:

- neutral standing: `pose_1_neutral.obj`;
- left-arm variation: `pose_2_left_arm.obj`; and
- right-arm variation: `pose_3_right_arm.obj`.

The outputs are stored in `outputs/pose_experiment/` and viewed together with `scripts/view_pose_experiment.py`. This establishes the conceptual distinction between:

- `beta`: body shape; and
- `theta`: body pose.

These experiments control model parameters directly. They do not estimate a real person's parameters from RGB images.

## SMPL-X Model and Parameters

SMPL-X is a parametric 3D human body model with expressive body, hand, face, and eye components. A simplified notation for its output is:

```text
M(beta, theta, psi)
```

where:

- `beta` represents learned body shape directions;
- `theta` represents body pose, including body joint rotations; and
- `psi` represents facial expression parameters.

The current scripts use ten body shape coefficients, zero expression, zero hand pose, zero jaw and eye pose, and controlled body pose values. Future personalization will require estimating suitable parameters from observations of a particular person and validating the resulting geometry against measurements or multi-view evidence.

## Project Structure

```text
Personalized3D-VTON/
├── README.md
├── .gitignore
├── data/
│   ├── raw/                  # Input data; do not commit private datasets
│   ├── frames/               # Extracted video frames
│   └── processed/            # Preprocessed data
├── models/
│   ├── README.md             # SMPL-X download and license instructions
│   └── smplx/
│       └── official/
│           └── smplx/        # Local SMPL-X files
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
├── notebooks/                # Reserved for exploratory notebooks
├── outputs/                  # Generated experiment meshes and results
│   ├── shape_experiment/
│   ├── all_betas/
│   └── pose_experiment/
└── requirements/
    └── environment.txt
```

The repository also contains local `.vscode/` configuration and downloaded model/output artifacts. These are development artifacts, not research source code.

## Installation

The following instructions describe the tested Conda environment. Do not recreate an existing working environment unless necessary.

### 1. Create and activate the Conda environment

```powershell
conda create -n human3d python=3.11 -y
conda activate human3d
```

### 2. Install dependencies

Install the PyTorch CUDA build appropriate for the machine using the official PyTorch selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). For the currently verified build, the command is:

```powershell
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install opencv-python numpy open3d trimesh smplx
```

Package versions verified in this project are listed in [requirements/environment.txt](requirements/environment.txt). A separate CUDA Toolkit installation is not required when the PyTorch package provides the required CUDA runtime.

### 3. Download SMPL-X model files

The official SMPL-X weights are not redistributed in this repository. Download them from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) and follow the applicable license and download terms.

Place the files in this local directory:

```text
models/smplx/official/smplx/
├── SMPLX_NEUTRAL.npz
├── SMPLX_MALE.npz
└── SMPLX_FEMALE.npz
```

The `.npz` model files are large and license-restricted. Do not upload them to GitHub. See [models/README.md](models/README.md) for the same model setup instructions.

## Verify the Environment and GPU

From the repository root:

```powershell
python scripts\test.py
```

The direct CUDA check is:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected current result includes `True` and `NVIDIA GeForce RTX 4050 Laptop GPU`.

## Running the Experiments

Run commands from the repository root. Generation scripts create directories under `outputs/`. Visualization scripts open interactive Open3D windows and require a graphical desktop session.

```powershell
python scripts\test_smplx.py
python scripts\view_smplx.py

python scripts\explore_shape.py
python scripts\view_shape_experiment.py

python scripts\explore_all_betas.py
python scripts\view_all_betas.py

python scripts\explore_pose.py
python scripts\view_pose_experiment.py
```

Run `scripts/test.py` separately when checking the environment. The scripts currently use the repository path `D:\Personalized3D-VTON` internally, so update that path if the project is moved to another location.

## Research Roadmap

### Stage 1 - SMPL-X Fundamentals: Completed

Model loading, CUDA execution, neutral mesh generation, Open3D visualization, independent beta experiments, and controlled pose experiments have been completed.

### Stage 2 - Single-Image 3D Human Estimation: Planned / Next

Evaluate an established monocular human reconstruction method such as 4DHumans/HMR2 or PARE and study how its output can initialize SMPL-X parameters.

### Stage 3 - Anthropometric Measurement Extraction: Planned

Define measurements, derive them from reconstructed vertices or joints, and compare them against reference measurements where available.

### Stage 4 - Multi-View RGB Capture and Body-Shape Optimization: Planned

Capture synchronized or carefully aligned views, estimate pose, and optimize body shape against multiple observations rather than a single image.

### Stage 5 - High-Fidelity Personalized Avatar: Planned

Evaluate shape consistency, pose consistency, surface quality, and measurement accuracy for a selected subject.

### Stage 6 - 3D Garment Representation: Planned

Represent garments as 3D meshes or another suitable structured representation with garment-specific dimensions and material properties.

### Stage 7 - Garment-Body Interaction and Cloth Deformation: Planned

Study fitting, collision handling, stretching, hanging, folds, and deformation for loose and tight garments across body shapes and poses.

### Stage 8 - Pose-Aware Virtual Try-On: Planned

Transfer garments onto the personalized avatar and render the result under changed poses and viewpoints.

### Stage 9 - Temporal Consistency: Planned

Measure stability across video frames and reduce jitter, flicker, and inconsistent garment motion.

### Stage 10 - Real-Time Optimization: Planned

Profile the pipeline, reduce latency and memory use, and evaluate whether a mirror-like experience is practical on the target hardware.

## Research Questions

1. How accurately can SMPL-X body shape parameters be estimated from RGB imagery?
2. How much does multi-view capture improve personalized body reconstruction compared with a single image?
3. How accurately can anthropometric measurements be extracted from a reconstructed body?
4. How can a personalized body representation improve virtual clothing fit?
5. How can garment deformation be modeled for different body shapes and poses?
6. How can the system maintain temporal consistency for real-time video?
7. What accuracy and latency trade-offs are acceptable for a practical virtual try-on system?

## Limitations and Scope

- RGB images do not guarantee exact physical body geometry.
- SMPL-X is a learned parametric representation, not a direct physical scan.
- The initial experiments use a neutral body model and manually controlled parameters.
- Current experiments are parameter-control experiments, not real-person reconstruction.
- No single-image estimator, multi-view optimizer, or anthropometric validation module is implemented yet.
- Garment reconstruction, cloth simulation, pose-aware VTON, temporal consistency, and real-time operation are not implemented yet.
- Model weights, private photographs, private datasets, and large generated outputs must not be committed.

## Reproducibility and Data Policy

Keep downloaded model weights in the local `models/` directory and keep raw or personal data under `data/raw/`. Do not commit personal photographs, private datasets, or restricted model files. Generated meshes and experiment outputs should be reviewed before committing because they can be large and are not source code. Record new experiments with their parameter values, input data description, software versions, and output location.

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
- [DPIDM and related 3D human research](https://github.com/tencent-ailab/3dgrids)

These references provide related body reconstruction, human parsing, image-based try-on, and 3D human modeling directions. They are research context, not claims that these systems are already integrated into this repository.

## Author

**Akhila Nisal Wedamestrige**

University of Moratuwa
Department of Electronic and Telecommunication Engineering

- GitHub: [https://github.com/AkhilaNisal](https://github.com/AkhilaNisal)
- Portfolio: [https://akhilanisal.github.io/akhila-portfolio/](https://akhilanisal.github.io/akhila-portfolio/)
