"""Fine-tune TerraMind on Sentinel-1 SAR. Settings are in config.yaml."""
import argparse
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml')
args = parser.parse_args()

subprocess.run(['terratorch', 'fit', '-c', str((root / args.config).resolve())],
               cwd=root, check=True)
