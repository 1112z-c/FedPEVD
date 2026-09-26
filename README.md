

## Installation

Python 3.11 and a CUDA-capable PyTorch installation are recommended. Install a PyTorch build compatible with the host CUDA driver if the plain wheel in `requirements.txt` is unsuitable.

Download `microsoft/codebert-base` separately. The checkpoint is not included in this archive.

## Run one dataset/seed

```bash
python run_primary.py \
  --dataset chrome \
  --seed 45 \
  --encoder_path /path/to/codebert-base \
  --output_dir outputs/seed51/chrome
```

The public entry point performs the exact required sequence: local reference training, deterministic canonical-head initialization, then Primary federated training. A successful run ends with `primary/comparison.json`.

To inspect commands and paths without training:

```bash
python run_primary.py --dataset chrome --seed 45 --encoder_path /path/to/codebert-base --output_dir outputs/dry --dry-run
```


## Data layout

Each dataset directory contains `client_0`, `client_1`, and `client_2`, each with `train.jsonl`, `valid.jsonl`, and `test.jsonl`. `SPLIT_MANIFEST.json` records the frozen client-first 20/10/70 split, and `SPLIT_MANIFEST.sha256` authenticates the manifest. Run `python verify_package.py` before training.

## Scope of `src`

`src` contains the exact internal runtime dependency closure used by Primary. These files are not additional public experiment arms; the only supported paper entry point is `run_primary.py`.

## Reproducibility boundary

The package freezes code, data split, hyperparameters, and selection protocol. GPU kernels and library builds can still introduce small numerical differences across hardware. The archive does not bundle pretrained CodeBERT weights or previously trained checkpoints.
