"""Evaluate a trained checkpoint and report the main segmentation metrics."""
import argparse
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml')
parser.add_argument('--checkpoint', help='Path to a .ckpt file (optional)')
args = parser.parse_args()

if args.checkpoint:
    ckpt = (root / args.checkpoint).resolve()
else:
    candidates = list((root / 'output').rglob('best-loss.ckpt'))
    if not candidates:
        parser.error('No best-loss.ckpt found under output/. Run training first.')
    ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
if not ckpt.is_file():
    parser.error(f'Checkpoint not found: {ckpt}')

print(f'Evaluating checkpoint: {ckpt}', flush=True)
cmd = ['terratorch', 'test', '-c', str((root / args.config).resolve()),
       '--ckpt_path', str(ckpt)]
result = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
log = root / 'output' / 'test_full.log'
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text(result.stdout)
if result.returncode:
    print(result.stdout[-6000:])
    raise SystemExit(result.returncode)

# Report selected metrics; save TerraTorch's full output above.
clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', result.stdout)
for metric, label in [('test/mIoU', 'mIoU'),
                      ('test/IoU_Water', 'Water IoU'),
                      ('test/F1_Score', 'F1 score')]:
    values = re.findall(rf'{re.escape(metric)}\s*[│|:]\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)', clean)
    if values:
        print(f'{label}: {float(values[-1]):.2%}')
    else:
        print(f'{label}: not found in log')
print(f'Full evaluation log: {log}')
