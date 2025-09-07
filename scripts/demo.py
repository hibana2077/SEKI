"""
SEKI 快速演示腳本
展示基本功能和使用方法
"""

import numpy as np
import sys
import os

# 添加項目根目錄到路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print("SEKI 快速演示")
print("="*50)

# 檢查依賴
missing_deps = []

try:
    import torch
    print("✓ PyTorch 可用")
except ImportError:
    missing_deps.append("torch")
    print("✗ PyTorch 不可用")

try:
    import numpy as np
    print("✓ NumPy 可用")
except ImportError:
    missing_deps.append("numpy")
    print("✗ NumPy 不可用")

try:
    from sklearn.datasets import make_circles
    print("✓ scikit-learn 可用")
except ImportError:
    missing_deps.append("scikit-learn")
    print("✗ scikit-learn 不可用")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib 可用")
except ImportError:
    missing_deps.append("matplotlib")
    print("✗ Matplotlib 不可用")

if missing_deps:
    print(f"\n缺少依賴: {', '.join(missing_deps)}")
    print("請運行: pip install -r requirements.txt")
    sys.exit(1)

# 嘗試導入 SEKI 模組
print("\n檢查 SEKI 模組...")
try:
    import seki
    print("✓ SEKI 主包可用")
    
    # 測試各個子模組
    modules_status = {}
    
    try:
        from seki.augmentation import seki_augmentation, mixup_augmentation
        modules_status['augmentation'] = True
        print("  ✓ 增強模組可用")
    except Exception as e:
        modules_status['augmentation'] = False
        print(f"  ✗ 增強模組不可用: {e}")
    
    try:
        from seki.models import StandardMLP
        modules_status['models'] = True
        print("  ✓ 模型模組可用")
    except Exception as e:
        modules_status['models'] = False
        print(f"  ✗ 模型模組不可用: {e}")
    
    try:
        from seki.evaluation import evaluate_accuracy
        modules_status['evaluation'] = True
        print("  ✓ 評估模組可用")
    except Exception as e:
        modules_status['evaluation'] = False
        print(f"  ✗ 評估模組不可用: {e}")
    
    try:
        from seki.training import TrainingConfig
        modules_status['training'] = True
        print("  ✓ 訓練模組可用")
    except Exception as e:
        modules_status['training'] = False
        print(f"  ✗ 訓練模組不可用: {e}")

except ImportError as e:
    print(f"✗ SEKI 主包不可用: {e}")
    modules_status = {k: False for k in ['augmentation', 'models', 'evaluation', 'training']}

# 簡單演示
print("\n" + "="*50)
print("運行簡單演示...")

if all(modules_status.values()):
    print("所有模組可用，運行完整演示...")
    
    try:
        # 生成簡單數據
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
        print(f"✓ 生成數據: {X.shape}")
        
        # 測試增強
        from seki.augmentation import seki_augmentation, mixup_augmentation
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        X_mixup, y_mixup = mixup_augmentation(X_tensor, y_tensor, alpha=1.0)
        print(f"✓ Mixup 增強: {X_mixup.shape}")
        
        X_seki, y_seki = seki_augmentation(X_tensor, y_tensor, alpha=1.0)
        print(f"✓ SEKI 增強: {X_seki.shape}")
        
        # 測試模型
        from seki.models import StandardMLP
        model = StandardMLP(input_dim=2, hidden_dim=16, output_dim=1)
        print(f"✓ 創建模型: {model.__class__.__name__}")
        
        # 測試前向傳播
        output = model(X_tensor[:10])
        print(f"✓ 模型輸出: {output.shape}")
        
        print("\n完整演示成功！")
        
    except Exception as e:
        print(f"演示中出現錯誤: {e}")
        print("請檢查依賴安裝")

else:
    print("部分模組不可用，運行基礎演示...")
    
    try:
        # 基礎 PyTorch 演示
        import torch
        import torch.nn as nn
        
        # 簡單模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                return self.sigmoid(self.linear(x))
        
        model = SimpleModel()
        x = torch.randn(10, 2)
        output = model(x)
        print(f"✓ 基礎模型測試成功: {output.shape}")
        
        # 簡單的 Mixup 實現
        def simple_mixup(x1, x2, alpha=1.0):
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
            return lam * x1 + (1 - lam) * x2
        
        x1 = torch.randn(5, 2)
        x2 = torch.randn(5, 2)
        mixed = simple_mixup(x1, x2)
        print(f"✓ 簡單 Mixup 測試成功: {mixed.shape}")
        
        print("\n基礎演示成功！")
        
    except Exception as e:
        print(f"基礎演示失敗: {e}")

# 項目結構檢查
print("\n" + "="*50)
print("檢查項目結構...")

required_dirs = ['seki', 'experiments', 'scripts', 'results']
for directory in required_dirs:
    dir_path = os.path.join(project_root, directory)
    if os.path.exists(dir_path):
        print(f"✓ {directory}/ 目錄存在")
    else:
        print(f"✗ {directory}/ 目錄不存在")

required_files = [
    'seki/__init__.py',
    'seki/augmentation.py', 
    'seki/models.py',
    'seki/evaluation.py',
    'seki/training.py',
    'requirements.txt'
]

for file_path in required_files:
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        print(f"✓ {file_path} 存在")
    else:
        print(f"✗ {file_path} 不存在")

# 使用建議
print("\n" + "="*50) 
print("使用建議:")
print("1. 安裝依賴: pip install -r requirements.txt")
print("2. 運行設置: python scripts/setup.py")
print("3. 運行實驗: python experiments/circle_experiment.py")
print("4. 查看文檔: docs/README.md")

if not all(modules_status.values()):
    print("\n注意: 某些 SEKI 模組不可用，可能是因為:")
    print("- 缺少必要的依賴包")
    print("- Python 環境路徑問題")
    print("- 模組導入錯誤")
    print("請檢查上述錯誤信息並安裝相應依賴")

print("\n演示完成！")
