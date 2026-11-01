"""Download model weights to the project model directory."""
import argparse
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3.5-4B')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    from huggingface_hub import snapshot_download
    root = Path(__file__).resolve().parents[1]
    snapshot_download(a.model, local_dir=a.out or root/'models'/a.model.split('/')[-1])
