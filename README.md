# Depression Recognition

Multimodal depression recognition code that fuses audio features (MFCC + Prosody) and facial Action Units (AU) with BiLSTM and cross-modal fusion for binary classification.

## Files

| File | Description |
|------|-------------|
| `model.py` | Dataset, model, training, and validation pipeline |
| `sample_data.py` | Single-sample feature loading and segmentation (AU stride=100) |
| `analysis.py` | Inference with a trained model on a single sample (AU stride=75) |
| `best_model.pth` | Pretrained model weights |

## Requirements

- Python 3
- PyTorch
- NumPy / Pandas
- scikit-learn

```bash
pip install torch numpy pandas scikit-learn
```

## Data Paths

Paths in this repository are **relative placeholders** and are not machine-specific.
Replace them with your local data locations before running.

Suggested layout:

```text
./data/
  processed/          # training pickles used by model.py
    data_dict_125.pkl
    mfcc_data_dict_125.pkl
    au_data_dict_125.pkl
    audio_data_dict_125.pkl
  au/                 # AU CSV files for single-sample scripts
  mfcc/               # MFCC CSV files
  egemaps/            # eGeMAPS / prosody CSV files
```

- Training (`model.py`): edit `load_data(data_dir=...)` or place pickles under `./data/processed/`
- Inference / inspection (`analysis.py`, `sample_data.py`): edit `data_dir`, `mfcc_dir`, and `audio_dir` at the bottom of each script

## Usage

**Training**:

```bash
python model.py --hidden_dim 128 --lr 1e-4 --epochs 50
```

**Single-sample inference** (also set `sample_id` in `analysis.py`):

```bash
python analysis.py
```

**Inspect segmented feature shapes**:

```bash
python sample_data.py
```

## Notes

- Default random seed `SEED=42`; validation split ratio 0.2 (stratified)
- Decision threshold: probability > 0.5 is predicted as depression
- `analysis.py` and `sample_data.py` use different AU segmentation strides; choose according to the intended use
