"""
SEKI 訓練模組
包含各種訓練策略和方法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Callable
import time
import os


class TrainingConfig:
    """訓練配置類"""
    
    def __init__(
        self,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        weight_decay: float = 1e-4,
        optimizer: str = 'adam',
        scheduler: Optional[str] = None,
        device: str = 'auto',
        save_best: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: bool = True
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = self._setup_device(device)
        self.save_best = save_best
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        
    def _setup_device(self, device: str) -> torch.device:
        """設置計算設備"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)


class EarlyStopping:
    """早停機制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        檢查是否應該早停
        
        Args:
            val_loss: 驗證損失
            model: 模型
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """保存模型檢查點"""
        self.best_weights = model.state_dict().copy()
    
    def restore_checkpoint(self, model: nn.Module):
        """恢復最佳模型權重"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def create_data_loaders(X_train, y_train, X_val=None, y_val=None, config: TrainingConfig = None):
    """
    創建 DataLoader
    
    Args:
        X_train: 訓練數據
        y_train: 訓練標籤
        X_val: 驗證數據
        y_val: 驗證標籤
        config: 訓練配置
        
    Returns:
        train_loader, val_loader (if validation data provided)
    """
    if config is None:
        config = TrainingConfig()
    
    # 轉換為 tensor
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        drop_last=False
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.FloatTensor(X_val)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.FloatTensor(y_val)
            
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            drop_last=False
        )
    
    return train_loader, val_loader


def setup_optimizer(model: nn.Module, config: TrainingConfig):
    """
    設置優化器和學習率調度器
    
    Args:
        model: 模型
        config: 訓練配置
        
    Returns:
        optimizer, scheduler
    """
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.learning_rate, 
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    scheduler = None
    if config.scheduler:
        if config.scheduler.lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif config.scheduler.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        elif config.scheduler.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    return optimizer, scheduler


def train_standard_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: Optional[DataLoader] = None,
    config: TrainingConfig = None,
    criterion: Optional[nn.Module] = None
) -> Dict:
    """
    標準模型訓練
    
    Args:
        model: 模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        config: 訓練配置
        criterion: 損失函數
        
    Returns:
        training_history: 訓練歷史
    """
    if config is None:
        config = TrainingConfig()
    
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    model = model.to(config.device)
    optimizer, scheduler = setup_optimizer(model, config)
    
    early_stopping = EarlyStopping(patience=config.patience) if config.early_stopping else None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = torch.sigmoid(output) > 0.5
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # 計算平均訓練指標
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 驗證階段
        val_loss = 0.0
        val_accuracy = 0.0
        
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(config.device), target.to(config.device)
                    output = model(data).squeeze()
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = torch.sigmoid(output) > 0.5
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            
            # 保存最佳模型
            if config.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
            
            # 早停檢查
            if early_stopping and early_stopping(val_loss, model):
                if config.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 學習率調度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader else avg_train_loss)
            else:
                scheduler.step()
        
        # 打印進度
        if config.verbose and (epoch + 1) % 10 == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config.epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            if val_loader:
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
    
    return history


def train_with_mixup(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: TrainingConfig = None,
    alpha: float = 1.0,
    mixup_prob: float = 0.5
) -> Dict:
    """
    使用 Mixup 增強的訓練
    
    Args:
        model: 模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        config: 訓練配置
        alpha: Mixup 的 alpha 參數
        mixup_prob: 使用 Mixup 的機率
        
    Returns:
        training_history: 訓練歷史
    """
    if config is None:
        config = TrainingConfig()
    
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(config.device)
    optimizer, scheduler = setup_optimizer(model, config)
    
    early_stopping = EarlyStopping(patience=config.patience) if config.early_stopping else None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            
            # 隨機決定是否使用 Mixup
            if np.random.rand() < mixup_prob:
                # Mixup 增強
                lam = np.random.beta(alpha, alpha)
                batch_size = data.size(0)
                index = torch.randperm(batch_size).to(config.device)
                
                mixed_data = lam * data + (1 - lam) * data[index]
                y_a, y_b = target, target[index]
                
                optimizer.zero_grad()
                output = model(mixed_data).squeeze()
                
                # Mixup 損失
                loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)
            else:
                # 標準訓練
                optimizer.zero_grad()
                output = model(data).squeeze()
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 計算準確率（使用原始標籤）
            if np.random.rand() >= mixup_prob:  # 只對非 Mixup 批次計算準確率
                pred = torch.sigmoid(output) > 0.5
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
        
        # 計算平均訓練指標
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / max(train_total, 1)  # 避免除零
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 驗證階段（與標準訓練相同）
        val_loss = 0.0
        val_accuracy = 0.0
        
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(config.device), target.to(config.device)
                    output = model(data).squeeze()
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = torch.sigmoid(output) > 0.5
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            
            # 早停檢查
            if early_stopping and early_stopping(val_loss, model):
                if config.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 學習率調度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader else avg_train_loss)
            else:
                scheduler.step()
        
        # 打印進度
        if config.verbose and (epoch + 1) % 10 == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config.epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            if val_loader:
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
    
    return history


def train_with_seki(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: TrainingConfig = None,
    seki_weight: float = 0.5,
    k_neighbors: int = 5
) -> Dict:
    """
    使用 SEKI 增強的訓練
    
    Args:
        model: 模型
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器
        config: 訓練配置
        seki_weight: SEKI 增強的權重
        k_neighbors: 近鄰數量
        
    Returns:
        training_history: 訓練歷史
    """
    from .augmentation import seki_augmentation
    
    if config is None:
        config = TrainingConfig()
    
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(config.device)
    optimizer, scheduler = setup_optimizer(model, config)
    
    early_stopping = EarlyStopping(patience=config.patience) if config.early_stopping else None
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config.device), target.to(config.device)
            
            # 應用 SEKI 增強
            try:
                augmented_data, augmented_labels = seki_augmentation(
                    data.cpu().numpy(), 
                    target.cpu().numpy(),
                    k=k_neighbors,
                    alpha=seki_weight
                )
                
                # 轉換回 tensor
                augmented_data = torch.FloatTensor(augmented_data).to(config.device)
                augmented_labels = torch.FloatTensor(augmented_labels).to(config.device)
                
                # 合併原始和增強數據
                combined_data = torch.cat([data, augmented_data], dim=0)
                combined_labels = torch.cat([target, augmented_labels], dim=0)
                
            except Exception as e:
                # 如果增強失敗，使用原始數據
                combined_data = data
                combined_labels = target
                if config.verbose and batch_idx == 0:
                    print(f"SEKI augmentation failed: {e}, using original data")
            
            optimizer.zero_grad()
            output = model(combined_data).squeeze()
            loss = criterion(output, combined_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 計算準確率（僅使用原始數據）
            original_output = output[:data.size(0)]
            pred = torch.sigmoid(original_output) > 0.5
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # 計算平均訓練指標
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 驗證階段
        val_loss = 0.0
        val_accuracy = 0.0
        
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(config.device), target.to(config.device)
                    output = model(data).squeeze()
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = torch.sigmoid(output) > 0.5
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            
            # 早停檢查
            if early_stopping and early_stopping(val_loss, model):
                if config.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 學習率調度
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader else avg_train_loss)
            else:
                scheduler.step()
        
        # 打印進度
        if config.verbose and (epoch + 1) % 10 == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch [{epoch+1}/{config.epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            if val_loader:
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            print(f"Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
    
    return history


def cross_validate(
    model_class: type,
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    config: TrainingConfig = None,
    model_kwargs: Dict = None,
    training_method: str = 'standard'
) -> Dict:
    """
    K 折交叉驗證
    
    Args:
        model_class: 模型類
        X: 輸入數據
        y: 標籤
        k_folds: 折數
        config: 訓練配置
        model_kwargs: 模型參數
        training_method: 訓練方法
        
    Returns:
        cv_results: 交叉驗證結果
    """
    from sklearn.model_selection import KFold
    
    if model_kwargs is None:
        model_kwargs = {}
    
    if config is None:
        config = TrainingConfig()
        config.verbose = False  # 交叉驗證時關閉詳細輸出
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}/{k_folds}...")
        
        # 分割數據
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 創建模型
        model = model_class(**model_kwargs)
        
        # 創建數據加載器
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, config)
        
        # 訓練模型
        if training_method == 'standard':
            history = train_standard_model(model, train_loader, val_loader, config)
        elif training_method == 'mixup':
            history = train_with_mixup(model, train_loader, val_loader, config)
        elif training_method == 'seki':
            history = train_with_seki(model, train_loader, val_loader, config)
        else:
            raise ValueError(f"Unknown training method: {training_method}")
        
        # 記錄結果
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': max(history['val_acc']),
            'best_val_loss': min(history['val_loss']),
            'final_val_acc': history['val_acc'][-1],
            'final_val_loss': history['val_loss'][-1],
            'history': history
        })
    
    # 計算統計結果
    val_accs = [result['best_val_acc'] for result in fold_results]
    val_losses = [result['best_val_loss'] for result in fold_results]
    
    cv_results = {
        'fold_results': fold_results,
        'mean_accuracy': np.mean(val_accs),
        'std_accuracy': np.std(val_accs),
        'mean_loss': np.mean(val_losses),
        'std_loss': np.std(val_losses),
        'config': config,
        'model_class': model_class.__name__
    }
    
    print(f"Cross-validation completed:")
    print(f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"Mean loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
    
    return cv_results


def save_model(model: nn.Module, filepath: str, additional_info: Dict = None):
    """
    保存模型
    
    Args:
        model: 模型
        filepath: 保存路徑
        additional_info: 附加信息
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'additional_info': additional_info or {}
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str) -> Dict:
    """
    加載模型
    
    Args:
        model: 模型實例
        filepath: 模型文件路徑
        
    Returns:
        additional_info: 附加信息
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    return checkpoint.get('additional_info', {})
