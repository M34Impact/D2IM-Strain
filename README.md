# D²IM-Strain

A deep learning framework for direct strain field prediction in bone from undeformed X-ray computed tomography (XCT) images, extending the original [D²IM](https://github.com/M34Impact/D2IM) framework.

For full details, please cite the following paper if using this code or data:

> **Evaluation of direct strain field prediction in bone with data-driven image mechanics (D²IM-Strain)**  
> Jon Valijonov, Peter Soar, James Le Houx, Gianluca Tozzi  
> *bioRxiv* (2026). DOI: [10.64898/2026.03.31.715417v1](https://doi.org/10.64898/2026.03.31.715417v1)

The original D²IM framework on which this work builds is described in:

> Soar P, Palanca M, Dall'Ara E, Tozzi G. Data-driven image mechanics (D2IM): A deep learning approach to predict displacement and strain fields from undeformed X-ray tomography images – Evaluation of bone mechanics. *Extreme Mech Lett* 2024; 71: 102202. DOI: [10.1016/j.eml.2024.102202](https://doi.org/10.1016/j.eml.2024.102202)

---

## Overview

Digital volume correlation (DVC) is the benchmark technique for full-field strain measurement in bone mechanics, but strain fields derived from numerical differentiation of displacement fields amplify high-frequency noise. D²IM-Strain proposes a direct strain prediction strategy that bypasses this differentiation step entirely.

Two prediction strategies are compared:
- **Displacement-derived strain** — strain computed by differentiating CNN-predicted displacement fields (original D²IM)
- **Direct strain prediction** — strain predicted directly from the undeformed XCT image and binary mask (D²IM-Strain)

Key results:
- Direct strain prediction achieves R² = 0.55 on the test set vs the displacement-derived approach
- Significantly reduced error in the pre-yield regime (strain < 10,000 µε, p < 0.01)
- 75% reduction in false-positive high-strain classifications (304 → 80)

---

## Model Architecture

The CNN architecture follows the original D²IM (VGGNet-inspired) to enable direct comparison:

- **Input**: 256×256 greyscale XCT slice + binary mask (2-channel input)
- **Encoder**: 4 convolutional blocks (3×3 kernels, ReLU, 2×2 max-pooling; filters: 32→64→128→256)
- **Dense layers**: 3 fully connected layers (512 units, ReLU)
- **Output**: 20×20 strain field (flattened), corresponding to DVC node spacing
- **Loss function**: Mean Absolute Error (MAE)
- **Optimiser**: Adam with staged learning rate schedule (1×10⁻³ → 1×10⁻⁴ → 1×10⁻⁵)
- **Regularisation**: Dropout (0.2) + L2 (1×10⁻⁶)
- **Training**: ~11 min 36 sec on NVIDIA RTX A6000 GPU (1,000 epochs)

---

## Dataset

The dataset used for this study is publicly available on Figshare:

> https://doi.org/10.6084/m9.figshare.25404220.v1

It consists of XCT scans of 10 porcine vertebrae (5 intact, 5 with artificial focal lesions) at 39 µm isotropic voxel size, loaded in two conditions (*in situ*). Ground-truth strain fields were computed using the open-source [SPAM](https://www.spam-project.dev/) Python library with a DVC window size of 50 voxels.

Tomograms were sliced into 2D cross-sections perpendicular to the loading axis (251 images after quality filtering), resized to 256×256 pixels. Data were split 60/20/20 (train/val/test), with slices from the same vertebra kept within a single split to avoid data leakage.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include Python, TensorFlow/Keras, NumPy, SciPy, and Matplotlib. SPAM is required for DVC ground-truth generation: https://www.spam-project.dev/

---

## Affiliations

Developed at the [University of Greenwich](https://www.gre.ac.uk/), Faculty of Engineering and Science.

- Centre for Advanced Simulation and Modelling, School of Computing and Mathematical Sciences
- Centre for Advanced Manufacturing and Materials, School of Engineering

Corresponding author: [g.tozzi@greenwich.ac.uk](mailto:g.tozzi@greenwich.ac.uk)

---

<p><small>Project based on the <a href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
