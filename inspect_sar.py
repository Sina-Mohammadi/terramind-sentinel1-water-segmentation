"""Inspect one Sen1Floods11 VV/VH SAR chip and its ground-truth mask.

Usage: python inspect_sar.py
Requires the Sen1Floods11 dataset from the README.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

DATA = Path("sen1floods11_v1.1/data")
files = sorted((DATA / "S1GRDHand").glob("*_S1Hand.tif"))
if not files:
    raise SystemExit("SAR files not found. Follow the dataset download steps in README.md")

# Only inspect chips that have a matching hand-labeled mask.
for image_path in files:
    mask_path = DATA / "LabelHand" / image_path.name.replace("_S1Hand.tif", "_LabelHand.tif")
    if mask_path.exists():
        break
else:
    raise SystemExit("No SAR chip has a matching hand-labeled mask.")

with rasterio.open(image_path) as src:
    if src.count != 2:
        raise ValueError(f"Expected VV/VH (2 bands), got {src.count} in {image_path}")
    vv, vh = src.read().astype("float32")
with rasterio.open(mask_path) as src:
    labels = src.read(1)

print(f"Chip: {image_path.name} | VV/VH units: dB | mask: {mask_path.name}")
for name, band in (("VV", vv), ("VH", vh)):
    valid = band[np.isfinite(band)]
    print(f"{name}: mean={valid.mean():.2f} dB, std={valid.std():.2f} dB")
print("Label values (-1=invalid, 0=other, 1=water):", np.unique(labels).tolist())

fig, axs = plt.subplots(1, 3, figsize=(11, 4))
for ax, band, name in zip(axs[:2], (vv, vh), ("VV", "VH")):
    valid = band[np.isfinite(band)]
    lo, hi = np.percentile(valid, [2, 98])
    ax.imshow(band, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(f"Sentinel-1 {name} (dB)")
axs[2].imshow(np.ma.masked_where(labels == -1, labels), cmap="Blues", vmin=0, vmax=1)
axs[2].set_title("Water mask (hand-labeled)")
for ax in axs:
    ax.set_axis_off()
fig.tight_layout()
out = Path("output/sar_chip.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(f"Saved {out}")
