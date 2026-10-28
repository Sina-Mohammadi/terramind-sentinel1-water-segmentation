# TerraMind Fine-Tuning for Sentinel-1 SAR Water Segmentation

This project fine-tunes the **TerraMind Tiny Earth observation foundation model** to identify water pixels in **Sentinel-1 synthetic aperture radar (SAR)** imagery. It uses the [Sen1Floods11](https://huggingface.co/datasets/blumenstiel/Sen1Floods11) dataset, which provides dual-polarization SAR image patches and hand-labeled water masks.

The objective is to adapt a pretrained geospatial model to a downstream remote-sensing task, then assess its water-segmentation performance on held-out imagery. The implementation uses **PyTorch and TerraTorch** to load the pretrained encoder, train a segmentation model, select a validation checkpoint, and evaluate predictions.

## Why Sentinel-1 SAR?

Unlike optical satellites, Sentinel-1 uses radar and can image the Earth's surface in darkness and through clouds. Water is often characterized by low backscatter because smooth surfaces reflect much of the incident radar signal away from the sensor. However, rough water, flooded vegetation, radar shadow, and other low-backscatter surfaces can complicate interpretation.

The model receives two radar channels:

- **VV:** vertically transmitted and vertically received radar backscatter.
- **VH:** vertically transmitted and horizontally received radar backscatter.

Using both polarizations gives the model complementary scattering information. The task is **water-versus-other semantic segmentation**: each valid pixel is classified as water or non-water. The dataset is designed for flood mapping, but the model does **not** compare pre- and post-flood images or directly estimate flood extent change.

## Dataset and preparation

The experiment uses the hand-labeled Sentinel-1 portion of **Sen1Floods11**:

- Input: two-band Sentinel-1 GRD image patches (`S1GRDHand`, VV and VH, in dB).
- Target: corresponding masks from `LabelHand` (`0 = Other`, `1 = Water`, `-1 = ignore`).
- Splits: the dataset's provided train, validation, and test file lists.
- Preprocessing: resize to **256 × 256** and standardize each channel using the means and standard deviations in `config.yaml` (VV: −12.599 / 5.195; VH: −20.293 / 5.890).
- Training augmentation: D4 (rotations and flips); validation and test use only resizing.

The project operates on **already-processed GRD image data**. It does not perform raw SAR radiometric calibration, speckle filtering, terrain correction, or interferometry.

## Model and training

`config.yaml` specifies the complete TerraTorch experiment:

1. **Encoder:** pretrained `terramind_v1_tiny`, configured for the `S1GRD` modality. Its weights are downloaded automatically on the first run.
2. **Segmentation model:** a TerraTorch encoder–decoder architecture with a `UNetDecoder` predicting the two classes.
3. **Fine-tuning:** the pretrained backbone and decoder are both trainable (`freeze_backbone: false`, `freeze_decoder: false`).
4. **Optimization:** Dice loss and AdamW with learning rate **2 × 10⁻⁵**, batch size **1**, and a maximum of **50 epochs**.
5. **Model selection:** save the checkpoint with the **lowest validation loss**, then evaluate it on the held-out test split.

Automatic validation-image plotting is disabled because the input has **two SAR channels**, rather than the three channels expected by the default RGB visualization. This does not affect training or metric computation.

## Test results

The latest completed evaluation produced the following **held-out test** scores:

| Metric | Result |
|---|---:|
| Mean Intersection over Union (mIoU) | **73.57%** |
| Water IoU | **55.25%** |
| F1 score | **83.48%** |
| Background (non-water) IoU | **91.89%** |

The model was trained and evaluated on the Sen1Floods11 dataset using TerraTorch. The results above are from the completed **50-epoch fine-tuning experiment**, evaluated on the held-out test split using the checkpoint selected by the lowest validation loss. Model weights and downloaded data are excluded from this repository.

The background class is called `Other` in the training configuration and includes valid non-water pixels. Water IoU is lower than the overall mIoU, so the water class remains the more challenging one. Pixels marked `-1` in the labels are ignored rather than treated as background. An initial three-epoch experiment obtained **57.03% mIoU** and **36.49% water IoU**. This is a comparison across fine-tuning runs, **not** a comparison against a non-pretrained or frozen-encoder baseline.

## Setup and execution

**Requirements:** Python 3.11+, enough free disk space for the model and dataset, and preferably a CUDA-capable GPU. Run the commands from the repository root.

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Download and extract Sen1Floods11

```bash
hf download blumenstiel/Sen1Floods11 sen1floods11_v1_1.tar.gz \
  --repo-type dataset --local-dir .
tar -xzf sen1floods11_v1_1.tar.gz
```

The expected directory structure is:

```text
sen1floods11_v1.1/
├── data/
│   ├── S1GRDHand/
│   └── LabelHand/
└── splits/
    ├── flood_train_data.txt
    ├── flood_valid_data.txt
    └── flood_test_data.txt
```

### 3. Examine a SAR image and its label

```bash
python inspect_sar.py
```

This displays VV and VH backscatter statistics and saves a visualization with the ground-truth mask to `output/sar_chip.png`.

### 4. Fine-tune TerraMind

```bash
python train.py
```

Training settings are defined in `config.yaml`. TerraTorch writes checkpoints under `output/sar_finetuned_50ep/`; the best checkpoint is selected using validation loss.

### 5. Evaluate

```bash
python evaluate.py
```

The script evaluates the most recently modified `best-loss.ckpt` found under `output/`, prints **mIoU, Water IoU, and F1**, and saves the complete TerraTorch output to `output/test_full.log`. To select a particular run explicitly:

```bash
python evaluate.py --config config.yaml --checkpoint path/to/best-loss.ckpt
```

When comparing multiple runs, specify the checkpoint explicitly to avoid evaluating the wrong experiment. Keep hyperparameter selection on the validation set and use the test set for final reporting.

## Repository files

| File | Purpose |
|---|---|
| `config.yaml` | Dataset paths, normalization, model, optimizer, training, and checkpoint settings |
| `train.py` | Launches TerraTorch fine-tuning using the configuration |
| `evaluate.py` | Runs held-out test evaluation and reports selected metrics |
| `inspect_sar.py` | Plots a Sentinel-1 VV/VH image patch and its labeled water mask |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes downloaded data, checkpoints, and generated outputs |

Dataset archives and trained weights are **not included** in the repository. If using a temporary directory on a shared server, copy any checkpoints or results you want to retain to persistent storage.

## References

- [TerraMind — IBM](https://github.com/IBM/terramind)
- [TerraTorch — geospatial deep learning framework](https://github.com/torchgeo/terratorch)
- [Sen1Floods11 dataset](https://huggingface.co/datasets/blumenstiel/Sen1Floods11)
