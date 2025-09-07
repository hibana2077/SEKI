"""
SEKI 數據增強方法
包含 SEKI, Mixup, Manifold Mixup, R-Mixup 等增強方法
"""

import numpy as np
import torch
import torch.nn.functional as F
import scipy.linalg as la


# --- 基礎幾何操作 ---

def to_polar(x):
    """將笛卡爾座標轉換為極座標 (r, theta)"""
    r = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    theta = torch.atan2(x[:, 1], x[:, 0])
    return r, theta


def to_cartesian(r, theta):
    """將極座標 (r, theta) 轉換為笛卡爾座標"""
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)


def slerp(p1, p2, t):
    """球面線性插值 (Slerp)"""
    omega = p2 - p1
    omega = torch.where(omega > np.pi, omega - 2 * np.pi, omega)
    omega = torch.where(omega < -np.pi, omega + 2 * np.pi, omega)
    return p1 + t * omega


# --- 主要增強方法 ---

def seki_augmentation(X, y, alpha=1.0):
    """
    SEKI (Symmetry-Equivariant Karcher Interpolation) 增強
    在圓流形上沿測地線進行插值
    
    Args:
        X: 輸入數據 [batch_size, 2]
        y: 標籤 [batch_size]
        alpha: Beta 分佈參數
        
    Returns:
        X_new: 增強後的數據
        y_new: 增強後的標籤
    """
    indices = torch.randperm(X.size(0))
    X1, y1 = X, y
    X2, y2 = X[indices], y[indices]
    
    lam = torch.tensor(np.random.beta(alpha, alpha, X.size(0)), 
                      device=X.device, dtype=X.dtype)
    
    # 轉換到極座標
    r1, theta1 = to_polar(X1)
    r2, theta2 = to_polar(X2)
    
    # 在流形上進行插值
    r_new = lam * r1 + (1 - lam) * r2  # 半徑線性插值
    theta_new = slerp(theta1, theta2, lam)  # 角度測地線插值
    
    # 轉換回笛卡爾座標
    X_new = to_cartesian(r_new, theta_new)
    y_new = lam * y1 + (1 - lam) * y2
    
    return X_new, y_new


def mixup_augmentation(X, y, alpha=1.0):
    """
    傳統 Mixup，在歐式空間中進行線性插值
    
    Args:
        X: 輸入數據
        y: 標籤
        alpha: Beta 分佈參數
        
    Returns:
        X_new: 混合後的數據
        y_new: 混合後的標籤
    """
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(X.size(0))
    
    X1, y1 = X, y
    X2, y2 = X[indices], y[indices]
    
    X_new = lam * X1 + (1 - lam) * X2
    y_new = lam * y1 + (1 - lam) * y2
    
    return X_new, y_new


def manifold_mixup_forward(model, X, y, mix_layer=1, alpha=1.0):
    """
    Manifold Mixup 前向傳播
    在指定隱層進行混合
    
    Args:
        model: 具有 feature_layers 和 classifier 的模型
        X: 輸入數據
        y: 標籤
        mix_layer: 混合的層索引
        alpha: Beta 分佈參數
        
    Returns:
        outputs: 模型輸出
        y_mixed: 混合後的標籤
    """
    lam = torch.tensor(np.random.beta(alpha, alpha), device=X.device, dtype=X.dtype)
    indices = torch.randperm(X.size(0))
    
    # 獲取到混合層的隱層表示
    h = X
    for i, layer in enumerate(model.feature_layers):
        h = layer(h)
        if i == mix_layer:
            # 在此層進行混合
            h_mixed = lam * h + (1 - lam) * h[indices]
            y_mixed = lam * y + (1 - lam) * y[indices]
            
            # 繼續前向傳播
            for j in range(i + 1, len(model.feature_layers)):
                h_mixed = model.feature_layers[j](h_mixed)
            
            return model.classifier(h_mixed), y_mixed
    
    # 如果沒有混合，正常前向傳播
    return model.classifier(h), y


# --- SPD 矩陣操作 (R-Mixup) ---

def make_spd_matrix(n_dim, n_samples, random_state=None):
    """
    生成 SPD (Symmetric Positive Definite) 矩陣
    
    Args:
        n_dim: 矩陣維度
        n_samples: 樣本數量
        random_state: 隨機種子
        
    Returns:
        matrices: SPD 矩陣數組 [n_samples, n_dim, n_dim]
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    matrices = []
    for _ in range(n_samples):
        A = np.random.randn(n_dim, n_dim)
        spd = A.T @ A + np.eye(n_dim)
        matrices.append(spd)
    
    return np.array(matrices)


def matrix_log(X):
    """矩陣對數 (用於 Log-Euclidean 幾何)"""
    eigenvals, eigenvecs = la.eigh(X)
    log_eigenvals = np.log(np.maximum(eigenvals, 1e-8))
    return eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T


def matrix_exp(X):
    """矩陣指數"""
    eigenvals, eigenvecs = la.eigh(X)
    exp_eigenvals = np.exp(eigenvals)
    return eigenvecs @ np.diag(exp_eigenvals) @ eigenvecs.T


def r_mixup_augmentation(X_spd, y, alpha=1.0):
    """
    R-Mixup: SPD 矩陣上的 Riemannian 插值
    
    Args:
        X_spd: SPD 矩陣 [n_samples, dim, dim]
        y: 標籤 [n_samples]
        alpha: Beta 分佈參數
        
    Returns:
        X_mixed: 混合後的 SPD 矩陣
        y_mixed: 混合後的標籤
    """
    lam = np.random.beta(alpha, alpha)
    indices = np.random.permutation(len(X_spd))
    
    X1, y1 = X_spd, y
    X2, y2 = X_spd[indices], y[indices]
    
    # Log-Euclidean 插值
    X_mixed = []
    for i in range(len(X1)):
        log_X1 = matrix_log(X1[i])
        log_X2 = matrix_log(X2[i])
        log_mixed = lam * log_X1 + (1 - lam) * log_X2
        X_mixed.append(matrix_exp(log_mixed))
    
    X_mixed = np.array(X_mixed)
    y_mixed = lam * y1 + (1 - lam) * y2
    
    return X_mixed, y_mixed


# --- 群等變操作 ---

def apply_rotation_2d(X, angle):
    """對 2D 數據應用旋轉變換"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], 
                                 device=X.device, dtype=X.dtype)
    return X @ rotation_matrix.T


def apply_reflection_2d(X, axis='x'):
    """對 2D 數據應用反射變換"""
    if axis == 'x':
        reflection_matrix = torch.tensor([[1, 0], [0, -1]], 
                                       device=X.device, dtype=X.dtype)
    else:  # y axis
        reflection_matrix = torch.tensor([[-1, 0], [0, 1]], 
                                       device=X.device, dtype=X.dtype)
    return X @ reflection_matrix.T


def group_equivariant_augmentation(X, y, group_size=8, alpha=1.0):
    """
    群等變數據增強
    對數據應用群變換後再進行插值
    
    Args:
        X: 輸入數據
        y: 標籤
        group_size: 群大小
        alpha: Beta 分佈參數
        
    Returns:
        X_new: 增強後的數據
        y_new: 增強後的標籤
    """
    # 先應用群變換
    angle = 2 * np.pi * np.random.randint(0, group_size) / group_size
    X_transformed = apply_rotation_2d(X, angle)
    
    # 然後應用標準增強
    return seki_augmentation(X_transformed, y, alpha)


# --- 輔助函數 ---

def get_augmentation_function(method_name):
    """根據方法名稱獲取增強函數"""
    augmentation_map = {
        'mixup': mixup_augmentation,
        'seki': seki_augmentation,
        'r_mixup': r_mixup_augmentation,
        'group_equivariant': group_equivariant_augmentation,
    }
    
    if method_name.lower() in augmentation_map:
        return augmentation_map[method_name.lower()]
    else:
        raise ValueError(f"Unknown augmentation method: {method_name}")


def validate_augmentation_input(X, y):
    """驗證增強函數的輸入"""
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor")
    if X.size(0) != y.size(0):
        raise ValueError("X and y must have the same batch size")
    
    return True
