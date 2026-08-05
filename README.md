# Depression Recognition

多模态抑郁症识别代码：融合音频（MFCC + Prosody）与面部动作单元（AU），基于 BiLSTM + 跨模态融合进行分类。

## 文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | 数据集、模型、训练与验证主流程 |
| `sample_data.py` | 单样本特征加载与分段（AU stride=100） |
| `analysis.py` | 加载已训练模型，对单样本做推理（AU stride=75） |
| `best_model.pth` | 预训练权重 |

## 环境依赖

- Python 3
- PyTorch
- NumPy / Pandas
- scikit-learn

```bash
pip install torch numpy pandas scikit-learn
```

## 使用方法

**训练**（需先将 `model.py` 中 `load_data()` 的 pkl 路径改为本地数据路径）：

```bash
python model.py --hidden_dim 128 --lr 1e-4 --epochs 50
```

**单样本推理**（修改 `analysis.py` 中的数据目录与 `sample_id`）：

```bash
python analysis.py
```

**查看分段特征形状**：

```bash
python sample_data.py
```

## 说明

- 默认随机种子 `SEED=42`，验证集比例 0.2（分层划分）
- 预测阈值：概率 > 0.5 判为抑郁
- `analysis.py` 与 `sample_data.py` 的 AU 分段步长不同，请按用途区分
