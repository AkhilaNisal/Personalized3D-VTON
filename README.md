# Personalized3D-VTON

**Personalized 3D Human Avatar for Real-Time Virtual Clothing Try-On**

Research project by **Akhila Nisal Wedamestrige**, University of Moratuwa, Department of Electronic and Telecommunication Engineering.

- GitHub: [AkhilaNisal/Personalized3D-VTON](https://github.com/AkhilaNisal/Personalized3D-VTON)
- Author profile: [https://github.com/AkhilaNisal](https://github.com/AkhilaNisal)
- Portfolio: [https://akhilanisal.github.io/akhila-portfolio/](https://akhilanisal.github.io/akhila-portfolio/)

## Project Overview

This project aims to develop a personalized and realistic virtual clothing try-on system. The long-term concept is that a person stands in front of a camera and slowly rotates while RGB images or video are captured from multiple viewpoints. These observations are used to estimate body shape and anthropometric proportions, personalize a generic parametric human model, and eventually fit garments to the resulting body.

The intended output is a **high-fidelity personalized 3D human avatar**, also described as a personalized 3D digital human model. This wording is deliberate: RGB-based reconstruction cannot guarantee an absolutely exact physical digital twin. Reconstruction quality must therefore be measured and validated rather than assumed.

### Current Milestone

The current milestone is the SMPL-X foundation and parameter-control stage:

```text
SMPL-X model loading
        |
Neutral mesh generation
        |
Shape parameter beta experiments
        |
Pose parameter theta experiments
        |
Mesh export and visualization
```

The next technical milestone is to select and test a lightweight human pose/body estimation method, then fit its observations to SMPL-X. The current repository does not yet reconstruct a real person from RGB input, perform multi-view optimization, simulate clothing, or provide real-time virtual try-on.

### Future Goal

The future system should support body-specific clothing fit, body motion, garment deformation, folds, loose and oversized clothing, tight clothing, temporal consistency, and eventually a real-time mirror-like interface. These are planned research objectives, not completed functionality.

## Current Research Direction

The planned architecture is:

```text
Multi-view RGB capture
        |
Person detection / segmentation
        |
Pose estimation
        |
Multi-view alignment
        |
SMPL-X reconstruction / fitting
        |
Body shape parameters beta
        |
Pose parameters theta
        |
Personalized 3D human avatar
        |
Anthropometric measurements
        |
Detailed appearance / surface reconstruction
        |
3D garment reconstruction and simulation
        |
Virtual try-on
```

The immediate research path is:

```text
Multi-view images/video -> SMPL-X -> personalized body -> anthropometric measurements
```

Clothing reconstruction, cloth simulation, pose-aware garment deformation, and real-time virtual try-on are later stages.

## Development Status

### Completed

| Item | Evidence or status |
| --- | --- |
| Project directory and Git repository | Repository exists locally with Git metadata and GitHub project reference |
| Conda environment | `human3d` environment is active and verified |
| Python | 3.11.16 |
| PyTorch and CUDA | PyTorch `2.11.0+cu128`; `torch.cuda.is_available()` is `True` |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU detected by PyTorch |
| Core libraries | OpenCV, NumPy, Open3D, trimesh, SciPy, and other documented packages verified |
| SMPL-X Python package | `smplx 0.1.28` installed and used successfully |
| Official SMPL-X model files | Downloaded separately and present locally; ignored by Git |
| Basic SMPL-X loading | Neutral model loaded successfully with CUDA |
| Default neutral mesh | Generated successfully with 10,475 vertices and 20,908 faces |
| OBJ export | Neutral mesh exported to `outputs/smplx_neutral.obj` |
| Open3D visualization | Generated mesh opened and visualized successfully |
| Single-beta shape experiment | beta 1 tested at five values |
| Ten-beta experiment | beta 1 through beta 10 tested independently |
| Shape visualization | Shape results visualized with Open3D |
| Pose experiment | Neutral, left-arm, and right-arm variations generated |
| Pose visualization | Pose results visualized together with Open3D |
| 4DHumans checkout | Repository cloned locally under `external/4D-Humans` |
| HMR2 package import | Local `hmr2` package import verified |
| HMR2 core setup | Core Python dependencies are installed in `human3d` |
| chumpy | `0.71` installed successfully |

### Current / Partially Prepared

- The 4DHumans/HMR2 setup was investigated as a possible monocular body-estimation baseline.
- The HMR2 package is available from the local editable checkout and imports successfully.
- HMR2 predicts SMPL rather than SMPL-X, so it is being considered as a possible baseline, not necessarily the final body model.
- The project is currently prioritizing a lightweight, understandable RGB-to-body-measurement-to-SMPL-X baseline.

### Not Completed

- Detectron2 is **not installed**.
- The official 4DHumans demo depends on Detectron2.
- The HMR2 model/checkpoint data was **not** fully downloaded.
- The large `hmr2_data.tar.gz` download was intentionally stopped and removed.
- HMR2 inference has **not** been verified.
- 4DHumans inference is **not** working or claimed as completed.
- No single-image body estimator is integrated into this project.
- No multi-view reconstruction or body-shape optimization is implemented.
- No anthropometric validation, garment reconstruction, cloth simulation, or real-time VTON is implemented.

## Verified Environment

The active `human3d` environment was inspected directly on Windows. The verified software versions are:

| Component | Verified value |
| --- | --- |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| CUDA available | `True` |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| OpenCV | 5.0.0 |
| NumPy | 2.4.6 |
| Open3D | 0.20.0 |
| trimesh | 5.1.0 |
| SMPL-X | 0.1.28 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| Pillow | 12.3.0 |
| tqdm | 4.70.1 |
| PyYAML | 6.0.3 |
| Jupyter | 1.1.1 |
| ipykernel | 7.3.0 |
| pytorch-lightning | 2.6.6 |
| scikit-image | 0.26.0 |
| einops | 0.8.2 |
| timm | 1.0.29 |
| dill | 0.4.1 |
| pandas | 3.0.6 |
| gdown | 6.4.0 |
| webdataset | 1.0.2 |
| yacs | 0.1.8 |
| chumpy | 0.71 |

The full environment record is in [requirements/environment.txt](requirements/environment.txt).

## Hardware and Software Environment

- Laptop: ASUS TUF F15 FX507ZU
- Operating system: Windows 11
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, approximately 6 GB VRAM
- Development distribution: Miniconda
- Conda environment: `human3d`
- Project path: `D:\Personalized3D-VTON`

## SMPL-X Model

SMPL-X, or Skinned Multi-Person Linear model with eXpressive hands and face, is a parametric 3D human body model. It is not a scan of a specific person. Instead, it provides a learned, parameterized representation that generates a human mesh from shape, pose, and expression parameters.

Conceptually, the generated mesh can be written as:

```text
M(beta, theta, psi)
```

where:

- `beta` (`β`) represents body shape;
- `theta` (`θ`) represents body and joint pose; and
- `psi` (`ψ`) represents facial expression.

The beta values are learned shape directions. They should not be described as directly corresponding to physical properties such as “fatness”, “chest size”, or “height”. Physical measurements must be estimated and validated separately.

The locally used official model files are:

```text
models/smplx/official/smplx/
├── SMPLX_FEMALE.npz
├── SMPLX_MALE.npz
└── SMPLX_NEUTRAL.npz
```

The official model weights are not redistributed by this repository. Obtain them from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) and comply with its license and download terms. The files are ignored by Git because of their size and licensing restrictions.

## Completed Experiments

### `scripts/test_smplx.py`

Loads the neutral SMPL-X model, selects CUDA when available, generates a default neutral human, and exports an OBJ mesh.

Verified result:

- vertices: `(10475, 3)`;
- faces: `(20908, 3)`; and
- output: `outputs/smplx_neutral.obj`.

### `scripts/view_smplx.py`

Loads `outputs/smplx_neutral.obj`, computes vertex normals, and visualizes the generated mesh using Open3D.

### `scripts/explore_shape.py`

Tests how changing one learned shape direction affects the generated body. Only beta 1 is changed, using:

```text
-3.0
-1.5
 0.0
+1.5
+3.0
```

The other shape and pose parameters remain at zero. Outputs are written to `outputs/shape_experiment/`.

### `scripts/view_shape_experiment.py`

Loads the five beta 1 meshes, separates them spatially, and visualizes the shape experiment using Open3D.

### `scripts/explore_all_betas.py`

Tests beta 1 through beta 10 independently. Each run sets one beta parameter to `+3.0` and keeps the other nine at zero. Outputs are written to `outputs/all_betas/`.

### `scripts/view_all_betas.py`

Loads the ten individual beta experiment meshes, arranges them in two rows, and visualizes them using Open3D.

### `scripts/explore_pose.py`

Demonstrates that pose can be changed independently from body shape. It generates:

- `pose_1_neutral.obj`;
- `pose_2_left_arm.obj`; and
- `pose_3_right_arm.obj`.

Outputs are written to `outputs/pose_experiment/`.

### `scripts/view_pose_experiment.py`

Loads the three pose meshes, arranges them horizontally, and visualizes the pose experiment using Open3D.

### `scripts/test.py`

Prints the installed versions of PyTorch, OpenCV, NumPy, Open3D, and trimesh, then reports CUDA availability and the detected GPU.

## 4DHumans / HMR2 Status

A local checkout of the 4DHumans repository is present at:

```text
external/4D-Humans
```

The local Python package was installed in editable mode, core dependencies were installed, and the `hmr2` package import was verified. The installed package metadata reports `hmr2 0.0.0` from the local checkout.

This is **not** a working HMR2 inference pipeline. The current status is:

| HMR2 item | Status |
| --- | --- |
| 4DHumans repository checkout | Completed locally |
| Editable package setup | Completed locally |
| Core Python dependency setup | Completed |
| `hmr2` package import | Verified |
| Detectron2 | Not installed |
| HMR2 checkpoint/model archive | Not fully downloaded |
| `hmr2_data.tar.gz` | Download intentionally stopped and removed |
| HMR2 inference | Not verified |
| Official 4DHumans demo | Not working / not claimed complete |

The official demo depends on Detectron2, which is missing in the current environment. A setup attempt also encountered a `pyrender`/OpenGL/EGL issue on Windows before model inference. This should not be interpreted as a failure of the RTX 4050 or CUDA installation: PyTorch CUDA and SMPL-X work correctly.

The HMR2 archive is not part of the normal setup instructions in this README. No large checkpoint download should be started automatically.

HMR2 predicts SMPL, while this project currently uses SMPL-X. HMR2 is therefore being considered as a possible baseline for body estimation, not as the final representation or an already integrated component.

## Current Model Strategy

The project will not depend immediately on a large model such as HMR2. The first goal is a lightweight and understandable baseline:

```text
RGB image/video
        |
Human pose or body landmarks
        |
Body measurements
        |
SMPL-X parameter fitting
        |
Personalized 3D body
```

A lightweight pose estimator such as MediaPipe Pose may be evaluated in future work, followed by optimization or fitting of SMPL-X. MediaPipe has not been installed or implemented in this repository and is only a candidate approach.

## Installation and Environment Setup

The current project uses the `human3d` Conda environment. The base environment can be created with:

```powershell
conda create -n human3d python=3.11 -y
conda activate human3d
```

The environment has already been prepared and verified locally. The major installed packages are recorded in [requirements/environment.txt](requirements/environment.txt), including PyTorch, OpenCV, NumPy, Open3D, trimesh, SMPL-X, SciPy, chumpy, and the packages used while investigating HMR2.

If Jupyter is needed in this environment, the verified kernel registration command is:

```powershell
python -m ipykernel install --user --name human3d --display-name "Python (human3d)"
```

The SMPL-X files must be obtained separately and placed in `models/smplx/official/smplx/`. The normal setup does not download the HMR2 archive, Detectron2, private datasets, or any other large model weights.

## Verification

Run the existing project smoke test from `D:\Personalized3D-VTON`:

```powershell
python scripts\test.py
```

A compact direct environment check is:

```powershell
python -c "import sys, torch, cv2, numpy, trimesh, smplx; print(sys.version); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__); print('trimesh:', trimesh.__version__); print('SMPL-X: imported')"
```

Verified result for the current environment:

```text
Python 3.11.16
PyTorch 2.11.0+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
OpenCV: 5.0.0
NumPy: 2.4.6
trimesh: 5.1.0
SMPL-X: imported
```

To verify the core SMPL-X result:

```powershell
python scripts\test_smplx.py
```

This currently produces a mesh with 10,475 vertices and 20,908 faces and exports `outputs/smplx_neutral.obj`.

## Windows Technical Notes

- Some 4DHumans code expects a `HOME` environment variable, which Windows does not always provide in the same way as Unix systems. The workaround used for the current PowerShell session was:

  ```powershell
  $env:HOME = $env:USERPROFILE
  ```

- SMPL-X currently runs successfully on the RTX 4050 through PyTorch CUDA.
- The HMR2 setup encountered a `pyrender`/OpenGL/EGL issue on Windows before model inference. This is separate from the working CUDA and SMPL-X setup.
- Detectron2 is not installed.
- The large HMR2 archive download was stopped because it was too large for the current workflow.

## Repository Structure

```text
Personalized3D-VTON/
├── README.md
├── SMPL-X.md
├── .gitignore
├── data/
│   ├── raw/                  # Raw/private input data; ignored by Git
│   ├── frames/               # Extracted frames; ignored by Git
│   └── processed/            # Preprocessed data; ignored by Git
├── models/
│   ├── README.md             # SMPL-X download and license instructions
│   └── smplx/
│       └── official/
│           └── smplx/        # Local SMPL-X model files; ignored by Git
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
├── notebooks/                # Notebook workspace
├── outputs/                  # Generated meshes; ignored by Git
├── external/
│   └── 4D-Humans/            # Local external checkout; ignored by Git
└── requirements/
    └── environment.txt      # Verified environment record
```

## Research Roadmap

### Phase 1 - SMPL-X Foundation: **COMPLETED**

- Install and verify the Python/CUDA environment.
- Obtain and load the official SMPL-X model.
- Generate and export a neutral mesh.
- Visualize the mesh with Open3D.
- Explore learned beta shape directions.
- Explore theta pose parameters.

### Phase 2 - Single-Image Body Estimation: **NEXT**

- Select a lightweight pose/body estimator.
- Test it on a real human image.
- Estimate body landmarks.
- Fit or optimize SMPL-X parameters.

### Phase 3 - Multi-View Reconstruction: **PLANNED**

- Capture multiple viewpoints while a person rotates.
- Detect and segment the person.
- Align the views.
- Estimate a consistent body shape.
- Optimize shared beta parameters across views.

### Phase 4 - Anthropometric Measurement: **PLANNED**

- Extract height, shoulder width, torso dimensions, limb lengths, and waist/hip-related measurements.
- Define the measurement protocol.
- Validate estimates against real measurements.

### Phase 5 - Detailed Personalized Avatar: **PLANNED**

- Improve surface geometry.
- Investigate texture and appearance.
- Add hair, head, or face detail if required by the research scope.

### Phase 6 - Garment Modeling: **PLANNED**

- Build a 3D garment representation.
- Fit garments to the personalized body.
- Model cloth deformation, folds, and body contact.
- Evaluate loose, oversized, and tight garments.

### Phase 7 - Real-Time Virtual Try-On: **PLANNED**

- Track pose in real time.
- Maintain temporal consistency.
- Use garment simulation or learned deformation.
- Build a camera or mirror interface.

## Research Questions

1. How accurately can SMPL-X body shape parameters be estimated from RGB imagery?
2. How much does multi-view capture improve personalized body reconstruction compared with a single image?
3. How accurately can anthropometric measurements be extracted from the reconstructed body?
4. How can a personalized body representation improve virtual clothing fit?
5. How can garment deformation be modeled for different body shapes and poses?
6. How can temporal consistency be maintained for real-time video?
7. What accuracy and latency trade-offs are acceptable for a practical virtual try-on system?

## Limitations

- RGB images do not guarantee exact physical body geometry.
- SMPL-X is a learned parametric representation, not a direct scan of a person.
- Current SMPL-X experiments use manually controlled parameters and a neutral body model.
- The current experiments do not reconstruct a real person from RGB imagery.
- HMR2 has not completed checkpoint setup or inference verification.
- Detectron2 is not installed.
- Garment reconstruction, cloth simulation, temporal consistency, and real-time operation are not implemented.
- Model weights, private photographs, private datasets, and large generated outputs must not be committed.

## Model File and Git Safety

The root `.gitignore` excludes new files matching:

```text
models/*
*.npz
*.pkl
*.pth
*.pt
*.ckpt
*.onnx
*.safetensors
outputs/
data/raw/
data/frames/
data/processed/
external/
```

It also excludes Python caches, virtual environments, Jupyter checkpoints, IDE settings, logs, and common generated mesh formats. `models/README.md` remains available for setup instructions while the downloaded SMPL-X weights remain ignored. The local `external/4D-Humans` checkout is also ignored because it is a separate external repository.

The current Git index already contains the generated OBJ meshes from the completed experiments. They were not removed in this documentation update because they are existing research artifacts and working outputs. No SMPL-X `.npz` weights, HMR2 archives, or private datasets are tracked. A future repository cleanup can remove the generated meshes from the index separately if a smaller source-only repository is desired.

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

These references provide research context. They are not claims that the listed systems are integrated or working in this repository.

## Current Status

The SMPL-X foundation and parameter experiments are working and verified on the RTX 4050. The current research direction is moving from controlled SMPL-X experiments toward a lightweight RGB human pose/body estimation baseline. The next technical milestone is selecting and testing that method, then fitting its observations to SMPL-X and validating the resulting personalized body with anthropometric measurements.
