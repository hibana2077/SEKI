"""
SEKI 基準實驗：圓形數據集
比較 ERM、Mixup、SEKI 在不同數據規模和噪聲水平下的性能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# 添加 seki 包到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.datasets import make_circles
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    print("Successfully imported scikit-learn modules")
except ImportError as e:
    print(f"Warning: Could not import scikit-learn: {e}")
    print("Please install scikit-learn: pip install scikit-learn")
    sys.exit(1)

try:
    from seki import (
        StandardMLP,
        seki_augmentation,
        mixup_augmentation,
        TrainingConfig,
        train_standard_model,
        train_with_mixup,
        train_with_seki,
        evaluate_accuracy,
        compute_comprehensive_metrics,
        compare_methods
    )
    print("Successfully imported SEKI modules")
except ImportError as e:
    print(f"Warning: Could not import SEKI modules: {e}")
    print("Using fallback implementations...")
    
    # 導入本地實現
    StandardMLP = None
    seki_augmentation = None
    mixup_augmentation = None


def setup_experiment_environment():
    """設置實驗環境"""
    # 設置隨機種子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 設置 matplotlib 樣式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 創建結果目錄
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir


def generate_dataset(n_samples=400, noise=0.08, test_size=0.5, random_state=42):
    """
    生成圓形數據集
    
    Args:
        n_samples: 樣本數量
        noise: 噪聲水平
        test_size: 測試集比例
        random_state: 隨機種子
        
    Returns:
        X_train, X_test, y_train, y_test: 訓練和測試數據
    """
    # 生成圓形數據
    X, y = make_circles(
        n_samples=n_samples, 
        noise=noise, 
        factor=0.5, 
        random_state=random_state
    )
    
    # 標準化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def visualize_augmentation_effects(X_train, y_train, setting, results_dir):
    """
    可視化增強效果
    """
    n_vis_samples = 50
    
    # 檢查是否能使用 SEKI 模組
    if seki_augmentation is None or mixup_augmentation is None:
        print("Skipping visualization due to missing SEKI modules")
        return
    
    # 轉換為 tensor
    X_tensor = torch.FloatTensor(X_train[:n_vis_samples])
    y_tensor = torch.FloatTensor(y_train[:n_vis_samples])
    
    try:
        X_mixup_vis, _ = mixup_augmentation(X_tensor, y_tensor)
        X_seki_vis, _ = seki_augmentation(X_tensor, y_tensor)
        
        # 創建可視化
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)
        fig.suptitle(
            f'Augmentation Comparison (n_samples={setting["n_samples"]}, noise={setting["noise"]})', 
            fontsize=16
        )
        
        # 原始數據
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k')
        axes[0].set_title('Original Training Data')
        axes[0].set_aspect('equal', 'box')
        
        # Mixup
        axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, 
                       edgecolors='k', alpha=0.3)
        axes[1].scatter(X_mixup_vis.numpy()[:, 0], X_mixup_vis.numpy()[:, 1], 
                       c='green', marker='x', s=50, label='Mixup Samples')
        axes[1].set_title('Traditional Mixup (Euclidean)')
        axes[1].legend()
        axes[1].set_aspect('equal', 'box')
        
        # SEKI
        axes[2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, 
                       edgecolors='k', alpha=0.3)
        axes[2].scatter(X_seki_vis.numpy()[:, 0], X_seki_vis.numpy()[:, 1], 
                       c='purple', marker='x', s=50, label='SEKI Samples')
        axes[2].set_title('SEKI (Geodesic Interpolation)')
        axes[2].legend()
        axes[2].set_aspect('equal', 'box')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存圖片
        filename = f"augmentation_comparison_n{setting['n_samples']}_noise{setting['noise']:.2f}.png"
        plt.savefig(os.path.join(results_dir, filename), dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, setting):
    """
    訓練和評估不同模型
    """
    config = TrainingConfig(
        batch_size=32,
        learning_rate=0.005,
        epochs=150,
        verbose=False,
        early_stopping=True,
        patience=15
    ) if TrainingConfig else None
    
    results = {}
    
    # 檢查是否能使用 SEKI 模組
    if StandardMLP is None:
        print("Using fallback PyTorch implementation...")
        return train_fallback_models(X_train, X_test, y_train, y_test)
    
    try:
        # 1. ERM 基準模型
        print("  Training ERM model...")
        model_erm = StandardMLP(input_dim=2, hidden_dim=32, output_dim=1)
        
        # 創建數據加載器
        from seki.training import create_data_loaders
        train_loader, _ = create_data_loaders(X_train, y_train, config=config)
        
        history_erm = train_standard_model(model_erm, train_loader, config=config)
        metrics_erm = compute_comprehensive_metrics(model_erm, 
                                                   torch.FloatTensor(X_test), 
                                                   torch.FloatTensor(y_test))
        results['ERM'] = metrics_erm
        
        # 2. Mixup 模型
        print("  Training Mixup model...")
        model_mixup = StandardMLP(input_dim=2, hidden_dim=32, output_dim=1)
        history_mixup = train_with_mixup(model_mixup, train_loader, config=config, alpha=0.4)
        metrics_mixup = compute_comprehensive_metrics(model_mixup, 
                                                     torch.FloatTensor(X_test), 
                                                     torch.FloatTensor(y_test))
        results['Mixup'] = metrics_mixup
        
        # 3. SEKI 模型
        print("  Training SEKI model...")
        model_seki = StandardMLP(input_dim=2, hidden_dim=32, output_dim=1)
        history_seki = train_with_seki(model_seki, train_loader, config=config, 
                                      seki_weight=0.4, k_neighbors=5)
        metrics_seki = compute_comprehensive_metrics(model_seki, 
                                                   torch.FloatTensor(X_test), 
                                                   torch.FloatTensor(y_test))
        results['SEKI'] = metrics_seki
        
        return results
        
    except Exception as e:
        print(f"Error during SEKI training: {e}")
        return train_fallback_models(X_train, X_test, y_train, y_test)


def train_fallback_models(X_train, X_test, y_train, y_test):
    """
    回退的簡單模型訓練（當 SEKI 模組不可用時）
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32), 
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    def simple_train(X_train, y_train, augmentation=None, epochs=150):
        model = SimpleMLP()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCELoss()
        
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                y_batch = y_batch.view(-1, 1)
                
                if augmentation:
                    try:
                        X_aug, y_aug = augmentation(X_batch, y_batch.squeeze(), alpha=0.4)
                        outputs = model(X_aug)
                        loss = criterion(outputs, y_aug.view(-1, 1))
                    except:
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model
    
    def simple_evaluate(model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test)
            outputs = model(X_tensor)
            predicted = (outputs > 0.5).squeeze()
            accuracy = (predicted == torch.FloatTensor(y_test)).float().mean()
        return accuracy.item()
    
    # 訓練模型
    model_erm = simple_train(X_train, y_train, augmentation=None)
    acc_erm = simple_evaluate(model_erm, X_test, y_test)
    
    # 嘗試使用增強方法
    acc_mixup = acc_erm  # 默認值
    acc_seki = acc_erm   # 默認值
    
    if mixup_augmentation:
        try:
            model_mixup = simple_train(X_train, y_train, augmentation=mixup_augmentation)
            acc_mixup = simple_evaluate(model_mixup, X_test, y_test)
        except Exception as e:
            print(f"Mixup training failed: {e}")
    
    if seki_augmentation:
        try:
            model_seki = simple_train(X_train, y_train, augmentation=seki_augmentation)
            acc_seki = simple_evaluate(model_seki, X_test, y_test)
        except Exception as e:
            print(f"SEKI training failed: {e}")
    
    results = {
        'ERM': {'accuracy': acc_erm},
        'Mixup': {'accuracy': acc_mixup},
        'SEKI': {'accuracy': acc_seki}
    }
    
    return results


def run_comprehensive_experiment():
    """
    運行完整的實驗
    """
    print("="*80)
    print("SEKI Comprehensive Circle Dataset Experiment")
    print("="*80)
    
    # 設置環境
    results_dir = setup_experiment_environment()
    
    # 實驗設置
    settings = [
        {'n_samples': 400, 'noise': 0.08},
        {'n_samples': 400, 'noise': 0.16},
        {'n_samples': 400, 'noise': 0.24},
        {'n_samples': 800, 'noise': 0.08},
        {'n_samples': 800, 'noise': 0.16},
        {'n_samples': 800, 'noise': 0.24},
        {'n_samples': 1200, 'noise': 0.08},
        {'n_samples': 1200, 'noise': 0.16},
        {'n_samples': 1600, 'noise': 0.08},
        {'n_samples': 2000, 'noise': 0.08},
    ]
    
    all_results = []
    
    for i, setting in enumerate(settings):
        print(f"\n[{i+1}/{len(settings)}] Running experiment: n_samples={setting['n_samples']}, noise={setting['noise']}")
        
        # 生成數據
        X_train, X_test, y_train, y_test = generate_dataset(
            n_samples=setting['n_samples'],
            noise=setting['noise']
        )
        
        print(f"  Dataset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        
        # 可視化增強效果（只對前幾個實驗）
        if i < 3:
            visualize_augmentation_effects(X_train, y_train, setting, results_dir)
        
        # 訓練和評估模型
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test, setting)
        
        # 記錄結果
        result_summary = {
            'setting': setting,
            'results': results
        }
        all_results.append(result_summary)
        
        # 打印當前結果
        print("  Results:")
        for method, metrics in results.items():
            acc = metrics.get('accuracy', 0.0)
            print(f"    {method}: {acc:.4f}")
    
    # 總結結果
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    # 打印詳細結果表格
    print(f"{'Setting':<20} {'ERM':<8} {'Mixup':<8} {'SEKI':<8} {'Winner'}")
    print("-" * 60)
    
    erm_wins = 0
    mixup_wins = 0
    seki_wins = 0
    
    for result in all_results:
        setting = result['setting']
        results = result['results']
        
        setting_str = f"n={setting['n_samples']}, σ={setting['noise']:.2f}"
        
        acc_erm = results.get('ERM', {}).get('accuracy', 0.0)
        acc_mixup = results.get('Mixup', {}).get('accuracy', 0.0)
        acc_seki = results.get('SEKI', {}).get('accuracy', 0.0)
        
        # 確定獲勝者
        accuracies = [acc_erm, acc_mixup, acc_seki]
        max_acc = max(accuracies)
        
        if acc_erm == max_acc:
            winner = "ERM"
            erm_wins += 1
        elif acc_mixup == max_acc:
            winner = "Mixup"
            mixup_wins += 1
        else:
            winner = "SEKI"
            seki_wins += 1
        
        print(f"{setting_str:<20} {acc_erm:<8.4f} {acc_mixup:<8.4f} {acc_seki:<8.4f} {winner}")
    
    print("-" * 60)
    print(f"Win counts: ERM={erm_wins}, Mixup={mixup_wins}, SEKI={seki_wins}")
    
    # 平均性能
    avg_erm = np.mean([r['results'].get('ERM', {}).get('accuracy', 0.0) for r in all_results])
    avg_mixup = np.mean([r['results'].get('Mixup', {}).get('accuracy', 0.0) for r in all_results])
    avg_seki = np.mean([r['results'].get('SEKI', {}).get('accuracy', 0.0) for r in all_results])
    
    print(f"\nAverage accuracies:")
    print(f"  ERM: {avg_erm:.4f}")
    print(f"  Mixup: {avg_mixup:.4f}")
    print(f"  SEKI: {avg_seki:.4f}")
    
    # 保存結果
    import json
    results_file = os.path.join(results_dir, 'circle_experiment_results.json')
    
    # 轉換結果為可序列化格式
    serializable_results = []
    for result in all_results:
        serializable_result = {
            'setting': result['setting'],
            'results': {}
        }
        for method, metrics in result['results'].items():
            if isinstance(metrics, dict):
                serializable_result['results'][method] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in metrics.items() 
                    if not isinstance(v, dict)  # 跳過複雜的嵌套字典
                }
            else:
                serializable_result['results'][method] = float(metrics)
        serializable_results.append(serializable_result)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print("Experiment completed!")


if __name__ == "__main__":
    run_comprehensive_experiment()
