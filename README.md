# SEKI (Symmetry-Equivariant Karcher Interpolation)

SEKI 是一個用於流形上對稱等變 Karcher 插值的 Python 包，專門用於數據增強和機器學習實驗。

## 項目結構

```
SEKI/
├── seki/                    # 核心包
│   ├── __init__.py         # 包初始化
│   ├── augmentation.py     # 數據增強方法
│   ├── models.py           # 神經網絡模型
│   ├── evaluation.py       # 評估指標
│   └── training.py         # 訓練工具
├── experiments/            # 實驗腳本
│   └── circle_experiment.py # 圓形數據集實驗
├── scripts/               # 工具腳本
│   ├── setup.py          # 項目設置
│   └── demo.py           # 快速演示
├── results/              # 實驗結果
├── tests/               # 單元測試
├── docs/               # 文檔
├── requirements.txt   # 依賴列表
└── README.md         # 項目主文檔
```

## 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 運行設置腳本
python scripts/setup.py
```

### 2. 快速演示

```bash
# 運行演示腳本檢查安裝
python scripts/demo.py
```

### 3. 運行實驗

```bash
# 運行圓形數據集實驗
python experiments/circle_experiment.py
```

## 核心功能

### 數據增強方法

- **SEKI**: 基於流形幾何的對稱等變 Karcher 插值
- **Mixup**: 傳統的歐幾里得空間線性插值
- **Manifold Mixup**: 隱藏層特徵空間的 Mixup
- **R-Mixup**: 針對 SPD 矩陣的黎曼 Mixup
- **Group Equivariant**: 群等變數據增強

### 模型架構

- **StandardMLP**: 標準多層感知機
- **ManifoldMixupMLP**: 支持 Manifold Mixup 的 MLP
- **GroupEquivariantMLP**: 群等變 MLP
- **SPDNet**: 針對 SPD 矩陣的網絡

### 評估指標

- **準確率評估**: 基礎分類性能
- **ECE/MCE**: 模型校準度評估
- **魯棒性測試**: 旋轉、反射、噪聲魯棒性
- **綜合指標**: 多維度性能評估

## 使用示例

### 基本使用

```python
import torch
from seki import (
    seki_augmentation, 
    mixup_augmentation,
    StandardMLP, 
    TrainingConfig,
    train_with_seki
)

# 準備數據
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100,)).float()

# 數據增強
X_aug, y_aug = seki_augmentation(X, y, alpha=1.0)

# 創建模型
model = StandardMLP(input_dim=2, hidden_dim=32, output_dim=1)

# 訓練配置
config = TrainingConfig(
    batch_size=32,
    learning_rate=0.001,
    epochs=100
)
```

## 依賴項

- **PyTorch** >= 1.9.0: 深度學習框架
- **NumPy** >= 1.21.0: 數值計算
- **scikit-learn** >= 1.0.0: 機器學習工具
- **Matplotlib** >= 3.5.0: 可視化
- **SciPy** >= 1.7.0: 科學計算

完整依賴列表請參考 `requirements.txt`。
