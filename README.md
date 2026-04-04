# AI for Science & Engineering: Flood Semantic Segmentation (Best Submission till Midnight Check-in: 0.1927)

This repository contains the inference and training methodology for our best-performing submission on the Phase 2 leaderboard (Flood IoU: 0.1927) as of the midnight check-in phase. 

## Overview

The objective of this challenge is to segment multi-modal satellite imagery into three distinct classes: **Land (0)**, **Flood Water (1)**, and **Permanent Water (2)**. Our highest scoring approach relies on a modified UNet++ architecture with a pre-trained ImageNet encoder, adapted to ingest multi-channel geospatial data.

### Notebook File
* `aise-phase2_1927.ipynb`: The end-to-end self-contained Kaggle notebook used to achieve the score.

---

## Technical Architecture

### 1. Model Configuration
* **Architecture:** UNet++
* **Encoder:** ResNet-101 (ImageNet pre-trained)
* **Input Adapter:** A custom learnable Convolutional Block Attention Module (CBAM) adapter. It takes the N-band satellite input and projects it into 3 channels to interface perfectly with the frozen early layers of the ResNet-101 encoder without losing spectral context.
* **Loss Function:** Focal Loss + Dice Loss with tailored class weights (heavily penalizing false negatives on the minority 'Flood' class).

### 2. Spectral Data Configuration
Through rigorous ablation, we found that dropping the SWIR band improved model performance by removing noise in the flood boundaries.
* **Input Channels (5 bands):** HH, HV (SAR) + Green, Red, NIR (Optical)
* **Normalization:** Image-wide Z-score normalization computed from the entire training corpus statistics.

### 3. Training Strategy
* **Data Starvation Mitigation:** The provided dataset contains only 59 training images. To prevent the 60M+ parameter ResNet-101 from memorizing the data, we employed strict augmentations including `CutMix`, `Horizontal/Vertical Flips`, `RandomRotate90`, and `GaussNoise`.
* **Reproducibility:** The notebook utilizes strict environment seeding. All PyTorch backend random generators (CUDA determinism, Python hash seeds) and Dataloader workers are locked to ensure the 0.1927 score is perfectly reproducible upon re-running.

---

## Post-Processing 

The primary difficulty of this task is the spectral indistinguishability between *Flood Water* and *Permanent Water*. Both surface types generate near-identical radar backscatter (SAR) and optical properties. 

Our pipeline handles this using probability thresholds. We extract raw softmax probabilities before the final `argmax` step, applying a **Flood-Priority Override**. If the model output registers the probability of Flood Water above a specific threshold, we force the output to Flood (1), even if the Permanent Water class had marginally higher activation. This significantly calibrated the boundary classifications where the baseline model hesitated.

---

## Setup and Execution

Due to Kaggle kernel caching with specific `tifffile` dependencies, there are two ways to reproduce this submission:

### Option A: Automated "Commit & Run" (Recommended)
If you wish to execute the notebook as a background Kaggle process:
1. Open the Kaggle Notebook Editor.
2. Navigate to **Add-ons -> Dependency Manager**.
3. Add the following line: `pip install -U imagecodecs` and click Save.
4. **Comment out** the very first cell in the notebook (`!pip install -U imagecodecs`).
5. Click **Save Version -> Save & Run All (Commit)**. The notebook will execute without kernel disruption and generate the `submission.csv`.

### Option B: Interactive Cell-by-Cell
If you are running the notebook interactively in a Kaggle session:
1. Run the very first cell: `!pip install -U imagecodecs`.
2. Navigate to the top menu and click **Run -> Restart Kernel and Clear Outputs**. (This step is mandatory to force Python to reload the newly installed codecs).
3. Do **not** run the first install cell again.
4. Run all subsequent cells one by one (or **Run -> Run All** starting from the imported modules cell).
5. The notebook will process the predictions and write the `submission.csv`.

*(Note: Ensure Kaggle Environment settings have internet access turned on if supplementary downloads are triggered.)*

---
## Results Checklist

- [x] Baseline UNet++ Training
- [x] CBAM Adapter implementation
- [x] Dataloader reproducibility improvements
- [x] Validation & Prediction extraction
- [x] Leaderboard Validation (0.1927)
