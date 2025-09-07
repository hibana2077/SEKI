"""
SEKI 神經網絡模型
包含標準 MLP, Manifold Mixup MLP, Group Equivariant MLP, SPD Net 等模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StandardMLP(nn.Module):
    """
    標準多層感知機
    
    Args:
        input_dim: 輸入維度
        hidden_dim: 隱層維度  
        output_dim: 輸出維度
    """
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()
        self.feature_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = x
        for layer in self.feature_layers:
            h = layer(h)
        return self.classifier(h)


class ManifoldMixupMLP(nn.Module):
    """
    支持 Manifold Mixup 的 MLP
    
    Args:
        input_dim: 輸入維度
        hidden_dim: 隱層維度
        output_dim: 輸出維度
    """
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1):
        super().__init__()
        self.feature_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, mix_layer=None, lam=None, indices=None):
        """
        前向傳播，支持 Manifold Mixup
        
        Args:
            x: 輸入張量
            mix_layer: 混合層索引
            lam: 混合係數
            indices: 混合索引
        """
        h = x
        
        for i, layer in enumerate(self.feature_layers):
            h = layer(h)
            
            # 在指定層進行 mixup
            if mix_layer is not None and i == mix_layer and lam is not None and indices is not None:
                h = lam * h + (1 - lam) * h[indices]
        
        return self.classifier(h)
    
    def get_hidden_features(self, x, layer_idx):
        """獲取指定層的隱層特徵"""
        h = x
        for i, layer in enumerate(self.feature_layers):
            h = layer(h)
            if i == layer_idx:
                return h
        return h


class GroupEquivariantMLP(nn.Module):
    """
    群等變 MLP (簡化版 G-CNN)
    
    Args:
        input_dim: 輸入維度
        hidden_dim: 隱層維度
        output_dim: 輸出維度
        group_size: 群大小
    """
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=1, group_size=8):
        super().__init__()
        self.group_size = group_size
        self.feature_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def apply_group_action(self, x, group_element):
        """
        應用群作用 (旋轉變換)
        
        Args:
            x: 輸入張量
            group_element: 群元素索引
        """
        if x.shape[1] == 2:  # 2D 數據
            angle = 2 * np.pi * group_element / self.group_size
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], 
                                         device=x.device, dtype=x.dtype)
            return x @ rotation_matrix.T
        return x
    
    def forward(self, x):
        """群等變前向傳播"""
        # 群等變處理：對所有群元素應用變換並平均
        outputs = []
        
        for g in range(self.group_size):
            x_g = self.apply_group_action(x, g)
            
            # 前向傳播
            h = x_g
            for layer in self.feature_layers:
                h = layer(h)
            output = self.classifier(h)
            outputs.append(output)
        
        # 群平均
        return torch.stack(outputs).mean(dim=0)


class SPDNet(nn.Module):
    """
    用於 SPD 矩陣的神經網路
    
    Args:
        matrix_dim: SPD 矩陣維度
        hidden_dim: 隱層維度
        output_dim: 輸出維度
    """
    def __init__(self, matrix_dim=3, hidden_dim=64, output_dim=1):
        super().__init__()
        self.matrix_dim = matrix_dim
        # SPD 矩陣的上三角元素數量
        input_dim = matrix_dim * (matrix_dim + 1) // 2  
        
        self.feature_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def spd_to_vector(self, spd_matrices):
        """
        將 SPD 矩陣轉換為向量 (提取上三角元素)
        
        Args:
            spd_matrices: SPD 矩陣 [batch_size, dim, dim]
            
        Returns:
            vectors: 向量表示 [batch_size, vector_dim]
        """
        if isinstance(spd_matrices, torch.Tensor):
            spd_matrices = spd_matrices.numpy()
            
        batch_size = spd_matrices.shape[0]
        vectors = []
        
        for i in range(batch_size):
            matrix = spd_matrices[i]
            # 提取上三角元素（包括對角線）
            triu_indices = np.triu_indices(self.matrix_dim)
            vector = matrix[triu_indices]
            vectors.append(vector)
        
        return torch.tensor(np.array(vectors), dtype=torch.float32)
    
    def forward(self, spd_matrices):
        """前向傳播"""
        # 將 SPD 矩陣轉換為向量
        x = self.spd_to_vector(spd_matrices)
        
        # 前向傳播
        h = x
        for layer in self.feature_layers:
            h = layer(h)
        
        return self.classifier(h)


class GroupEquivariantConv1D(nn.Module):
    """
    群等變 1D 卷積層 (用於序列數據)
    
    Args:
        in_channels: 輸入通道數
        out_channels: 輸出通道數
        kernel_size: 卷積核大小
        group_size: 群大小
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, group_size=8):
        super().__init__()
        self.group_size = group_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2)
        
    def apply_group_action(self, x, group_element):
        """應用群作用到序列數據"""
        # 簡化的群作用：循環移位
        shift = group_element
        return torch.roll(x, shifts=shift, dims=2)
        
    def forward(self, x):
        """群等變卷積前向傳播"""
        outputs = []
        
        for g in range(self.group_size):
            x_g = self.apply_group_action(x, g)
            conv_out = self.conv(x_g)
            outputs.append(conv_out)
        
        # 群平均
        return torch.stack(outputs).mean(dim=0)


# --- 模型工廠和工具函數 ---

def create_model(model_type, **kwargs):
    """
    根據類型創建模型
    
    Args:
        model_type: 模型類型
        **kwargs: 模型參數
        
    Returns:
        model: 創建的模型
    """
    model_map = {
        'standard': StandardMLP,
        'manifold_mixup': ManifoldMixupMLP,
        'group_equivariant': GroupEquivariantMLP,
        'spd': SPDNet,
    }
    
    if model_type in model_map:
        return model_map[model_type](**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """計算模型參數數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_shape):
    """
    模型摘要信息
    
    Args:
        model: PyTorch 模型
        input_shape: 輸入形狀
    """
    total_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Input shape: {input_shape}")
    
    # 計算模型大小（假設 float32）
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return total_params


def init_weights(model, init_type='xavier'):
    """
    初始化模型權重
    
    Args:
        model: PyTorch 模型
        init_type: 初始化類型
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def freeze_layers(model, layer_names):
    """
    凍結指定層的參數
    
    Args:
        model: PyTorch 模型
        layer_names: 要凍結的層名稱列表
    """
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False


def get_model_device(model):
    """獲取模型所在設備"""
    return next(model.parameters()).device
