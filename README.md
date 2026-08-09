# Depression Recognition (Journal Submission)

> **Note:** This repository is provided solely for journal manuscript submission (code re-upload / supplementary material). It is not intended as a maintained open-source project.

Multimodal depression detection model used in the manuscript: facial Action Units (AU) + acoustic features (MFCC + selected prosody descriptors), with bidirectional LSTM encoding, multi-head self-attention, cross-modal fusion, and gender-conditioned gating.

## Package layout

```text
src/multimodal_depression/
  constants.py              # feature dimensions
  config.py                 # ModelConfig hyperparameters
  models/
    layers.py               # TimeAlign, BiLSTM, attention, fusion layers
    classifier.py           # Baseline MultiModalDepressionClassifier
    ablation.py             # ablation variants
  utils/
    checkpoint.py           # BestModelSaver / save_best_model
scripts/
  smoke_test_baseline.py    # forward-pass and checkpoint utility smoke test
requirements.txt
```

Pretrained weight files (`*.pth`) are intentionally not included in this repository.

## Baseline forward path

```text
Audio (17-D) / AU (17-D)
  -> TimeAlign (audio length 500, AU length 125)
  -> BiLSTMBlock + LayerNorm
  -> MultiHeadAttention (8 heads)
  -> CrossModalAttention (bidirectional audio <-> AU)
       output: [f_a; f_a_tilde; f_v; f_v_tilde] in R^{4H}
  -> DynamicFusionLayer (gender-conditioned gating, emb_dim=32)
  -> MLP classifier + sigmoid
```

Default hyperparameters: `hidden_dim=128`, `num_heads=8`, `dropout=0.3`  
(fused dimension `4H = 512`).

## Keep best model parameters

```python
from multimodal_depression.utils import BestModelSaver

saver = BestModelSaver("best_model.pth", mode="max")  # maximize AUC / F1
for epoch in range(epochs):
    metrics = validate(...)
    improved = saver.update(model, metrics["auc"], extra={"epoch": epoch})
saver.load_best(model)
```

## Quick check

```bash
pip install -r requirements.txt
python scripts/smoke_test_baseline.py
```

## Ablation builders

| Name | Change relative to Baseline |
|------|-----------------------------|
| `Baseline` | Full model |
| `LegacyMean` | Remove multi-head self-attention |
| `NoTimeAlign` | Remove `TimeAlign` |
| `NoCrossModal` | Replace cross-modal fusion with concatenation |
| `ConcatFusion` | Replace gender gating with gender concatenation |
| `NoGender` | Zero gender input |

## Data paths

Feature tensors are expected as NumPy arrays with shapes compatible with the model input dimensions above. Place processed data under a local `./data/processed/` directory when running training. Paths referenced in local experiments are placeholders and are not machine-specific.
