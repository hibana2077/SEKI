"""
SEKI 評估指標和工具
包含準確率、ECE校準度、群變換魯棒性等評估方法
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def evaluate_accuracy(model, X_test, y_test, threshold=0.5):
    """
    評估模型準確率
    
    Args:
        model: 訓練好的模型
        X_test: 測試數據
        y_test: 測試標籤
        threshold: 分類閾值
        
    Returns:
        accuracy: 準確率
        probabilities: 預測機率
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'spd_to_vector'):  # SPD model
            outputs = model(X_test).squeeze()
        else:
            outputs = model(X_test).squeeze()
        
        predictions = (outputs > threshold).float()
        
        if isinstance(y_test, torch.Tensor):
            y_test_np = y_test.numpy()
        else:
            y_test_np = y_test
            
        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.numpy()
        else:
            predictions_np = predictions
            
        accuracy = (predictions_np == y_test_np).mean()
        
    return accuracy, outputs.numpy()


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    計算 Expected Calibration Error (ECE)
    衡量模型輸出機率的校準度
    
    Args:
        y_true: 真實標籤
        y_prob: 預測機率
        n_bins: 分桶數量
        
    Returns:
        ece: 期望校準誤差
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.numpy()
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(y_true)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在此 bin 中的樣本
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    """
    計算 Maximum Calibration Error (MCE)
    
    Args:
        y_true: 真實標籤
        y_prob: 預測機率
        n_bins: 分桶數量
        
    Returns:
        mce: 最大校準誤差
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.numpy()
        
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_errors = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            calibration_errors.append(np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return max(calibration_errors) if calibration_errors else 0.0


def rotation_robustness_test(model, X_test, y_test, n_rotations=8):
    """
    測試模型對旋轉變換的魯棒性
    只適用於 2D 數據
    
    Args:
        model: 訓練好的模型
        X_test: 測試數據
        y_test: 測試標籤
        n_rotations: 旋轉次數
        
    Returns:
        mean_accuracy: 平均準確率
        std_accuracy: 準確率標準差
    """
    if X_test.shape[1] != 2:
        # 對於非 2D 數據，返回原始準確率
        acc, _ = evaluate_accuracy(model, X_test, y_test)
        return acc, 0.0
    
    model.eval()
    accuracies = []
    
    for i in range(n_rotations):
        angle = 2 * np.pi * i / n_rotations
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], 
                                     device=X_test.device, dtype=X_test.dtype)
        
        X_rotated = X_test @ rotation_matrix.T
        acc, _ = evaluate_accuracy(model, X_rotated, y_test)
        accuracies.append(acc)
    
    return np.mean(accuracies), np.std(accuracies)


def reflection_robustness_test(model, X_test, y_test):
    """
    測試模型對反射變換的魯棒性
    只適用於 2D 數據
    
    Args:
        model: 訓練好的模型
        X_test: 測試數據
        y_test: 測試標籤
        
    Returns:
        mean_accuracy: 平均準確率
        std_accuracy: 準確率標準差
    """
    if X_test.shape[1] != 2:
        acc, _ = evaluate_accuracy(model, X_test, y_test)
        return acc, 0.0
    
    model.eval()
    accuracies = []
    
    # 原始數據
    acc_orig, _ = evaluate_accuracy(model, X_test, y_test)
    accuracies.append(acc_orig)
    
    # x 軸反射
    reflection_x = torch.tensor([[1, 0], [0, -1]], 
                               device=X_test.device, dtype=X_test.dtype)
    X_reflected_x = X_test @ reflection_x.T
    acc_x, _ = evaluate_accuracy(model, X_reflected_x, y_test)
    accuracies.append(acc_x)
    
    # y 軸反射
    reflection_y = torch.tensor([[-1, 0], [0, 1]], 
                               device=X_test.device, dtype=X_test.dtype)
    X_reflected_y = X_test @ reflection_y.T
    acc_y, _ = evaluate_accuracy(model, X_reflected_y, y_test)
    accuracies.append(acc_y)
    
    return np.mean(accuracies), np.std(accuracies)


def noise_robustness_test(model, X_test, y_test, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    """
    測試模型對噪聲的魯棒性
    
    Args:
        model: 訓練好的模型
        X_test: 測試數據
        y_test: 測試標籤
        noise_levels: 噪聲水平列表
        
    Returns:
        results: 各噪聲水平下的準確率字典
    """
    model.eval()
    results = {}
    
    for noise_level in noise_levels:
        noise = torch.randn_like(X_test) * noise_level
        X_noisy = X_test + noise
        acc, _ = evaluate_accuracy(model, X_noisy, y_test)
        results[noise_level] = acc
    
    return results


def compute_comprehensive_metrics(model, X_test, y_test):
    """
    計算綜合評估指標
    
    Args:
        model: 訓練好的模型
        X_test: 測試數據
        y_test: 測試標籤
        
    Returns:
        metrics: 包含各種指標的字典
    """
    # 基本準確率和機率
    accuracy, probabilities = evaluate_accuracy(model, X_test, y_test)
    
    # 校準度指標
    ece = expected_calibration_error(y_test, probabilities)
    mce = maximum_calibration_error(y_test, probabilities)
    
    # 魯棒性測試
    rot_acc_mean, rot_acc_std = rotation_robustness_test(model, X_test, y_test)
    ref_acc_mean, ref_acc_std = reflection_robustness_test(model, X_test, y_test)
    
    # 噪聲魯棒性
    noise_results = noise_robustness_test(model, X_test, y_test)
    
    # 其他指標
    if isinstance(y_test, torch.Tensor):
        y_test_np = y_test.numpy()
    else:
        y_test_np = y_test
        
    try:
        auc = roc_auc_score(y_test_np, probabilities)
    except:
        auc = 0.5  # 如果無法計算 AUC，使用默認值
    
    predictions = (probabilities > 0.5).astype(int)
    f1 = f1_score(y_test_np, predictions, average='binary', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'ece': ece,
        'mce': mce,
        'rotation_robustness_mean': rot_acc_mean,
        'rotation_robustness_std': rot_acc_std,
        'reflection_robustness_mean': ref_acc_mean,
        'reflection_robustness_std': ref_acc_std,
        'noise_robustness': noise_results
    }
    
    return metrics


def print_metrics(metrics, method_name):
    """
    格式化打印評估指標
    
    Args:
        metrics: 指標字典
        method_name: 方法名稱
    """
    print(f"\n=== {method_name} Results ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")
    print(f"MCE: {metrics['mce']:.4f}")
    print(f"Rotation Robustness: {metrics['rotation_robustness_mean']:.4f} ± {metrics['rotation_robustness_std']:.4f}")
    print(f"Reflection Robustness: {metrics['reflection_robustness_mean']:.4f} ± {metrics['reflection_robustness_std']:.4f}")
    
    print("Noise Robustness:")
    for noise_level, acc in metrics['noise_robustness'].items():
        print(f"  σ={noise_level}: {acc:.4f}")


def compare_methods(results_dict):
    """
    比較多個方法的性能
    
    Args:
        results_dict: 包含各方法結果的字典
    """
    print("\n" + "="*80)
    print("METHOD COMPARISON SUMMARY")
    print("="*80)
    
    # 準備比較表格
    methods = list(results_dict.keys())
    
    print(f"{'Method':<20} {'Accuracy':<10} {'ECE':<10} {'Rot_Robust':<12}")
    print("-" * 60)
    
    for method in methods:
        metrics = results_dict[method]
        acc = metrics['accuracy']
        ece = metrics['ece']
        rot_rob = metrics['rotation_robustness_mean']
        
        print(f"{method:<20} {acc:<10.4f} {ece:<10.4f} {rot_rob:<12.4f}")
    
    # 統計獲勝次數
    print(f"\n--- Performance Rankings ---")
    
    # 準確率排名
    acc_ranking = sorted(methods, key=lambda x: results_dict[x]['accuracy'], reverse=True)
    print(f"Accuracy ranking: {' > '.join(acc_ranking)}")
    
    # ECE 排名 (越小越好)
    ece_ranking = sorted(methods, key=lambda x: results_dict[x]['ece'])
    print(f"Calibration (ECE) ranking: {' > '.join(ece_ranking)}")
    
    # 旋轉魯棒性排名
    rot_ranking = sorted(methods, key=lambda x: results_dict[x]['rotation_robustness_mean'], reverse=True)
    print(f"Rotation robustness ranking: {' > '.join(rot_ranking)}")


def save_results_to_file(results_dict, filename):
    """
    將結果保存到文件
    
    Args:
        results_dict: 結果字典
        filename: 保存文件名
    """
    import json
    
    # 轉換 numpy 類型為 Python 原生類型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_numpy(results_dict)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {filename}")


def load_results_from_file(filename):
    """
    從文件加載結果
    
    Args:
        filename: 結果文件名
        
    Returns:
        results_dict: 結果字典
    """
    import json
    import os
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file {filename} not found")
    
    with open(filename, 'r', encoding='utf-8') as f:
        results_dict = json.load(f)
    
    return results_dict


def bootstrap_confidence_interval(metric_values, confidence_level=0.95, n_bootstrap=1000):
    """
    計算指標的 bootstrap 置信區間
    
    Args:
        metric_values: 指標值列表
        confidence_level: 置信水平
        n_bootstrap: bootstrap 樣本數
        
    Returns:
        lower_bound: 下界
        upper_bound: 上界
    """
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound


def statistical_significance_test(method1_results, method2_results, test_type='paired_t'):
    """
    統計顯著性檢驗
    
    Args:
        method1_results: 方法1的結果列表
        method2_results: 方法2的結果列表
        test_type: 檢驗類型
        
    Returns:
        p_value: p值
        is_significant: 是否顯著
    """
    from scipy import stats
    
    if test_type == 'paired_t':
        statistic, p_value = stats.ttest_rel(method1_results, method2_results)
    elif test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(method1_results, method2_results)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    is_significant = p_value < 0.05
    
    return p_value, is_significant
