# utils/scalers.py
import torch
import numpy as np
from typing import Union, Tuple, Optional

# class StandardScaler():
#     """标准化: (x - mean) / std，只对非0值进行标准化"""
#     def __init__(self, eps: float = 1e-8):
#         self.eps = eps
#         self.mean = None
#         self.std = None
        
#     def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
#         """计算非0值的均值和标准差"""
#         if isinstance(data, torch.Tensor):
#             # 创建非0值的mask
#             nonzero_mask = (data != 0)
#             # 只对非0值计算均值和标准差
#             nonzero_data = data[nonzero_mask]
#             self.mean = nonzero_data.mean()
#             self.std = nonzero_data.std()
#         else:
#             # 创建非0值的mask
#             nonzero_mask = (data != 0)
#             # 只对非0值计算均值和标准差
#             nonzero_data = data[nonzero_mask]
#             self.mean = np.mean(nonzero_data)
#             self.std = np.std(nonzero_data)
            
#     def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
#         """标准化变换，只转换非0值"""
#         if self.mean is None or self.std is None:
#             raise ValueError("Scaler must be fitted before transform")
            
#         result = data.clone() if isinstance(data, torch.Tensor) else data.copy()
#         nonzero_mask = (data != 0)
        
#         if isinstance(data, torch.Tensor):
#             result[nonzero_mask] = (data[nonzero_mask] - self.mean) / (self.std + self.eps)
#         else:
#             result[nonzero_mask] = (data[nonzero_mask] - self.mean) / (self.std + self.eps)
            
#         return result
            
#     def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
#         """逆变换，只转换非0值"""
#         if self.mean is None or self.std is None:
#             raise ValueError("Scaler must be fitted before inverse_transform")
            
#         result = data.clone() if isinstance(data, torch.Tensor) else data.copy()
#         nonzero_mask = (data != 0)
        
#         if isinstance(data, torch.Tensor):
#             result[nonzero_mask] = data[nonzero_mask] * self.std + self.mean
#         else:
#             result[nonzero_mask] = data[nonzero_mask] * self.std + self.mean
            
#         return result 

class StandardScaler():
    """标准化: (x - mean) / std
    fit时只用非0值计算均值和标准差
    transform时对所有值进行变换"""
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None
        self.std = None
        
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """计算每个特征非0值的均值和标准差"""
        n_features = data.shape[-1]
        
        if isinstance(data, torch.Tensor):
            self.mean = torch.zeros(n_features)
            self.std = torch.zeros(n_features)
            
            for i in range(n_features):
                feature_data = data[..., i]
                nonzero_mask = (feature_data != 0)
                if nonzero_mask.any():
                    nonzero_data = feature_data[nonzero_mask]
                    self.mean[i] = nonzero_data.mean()
                    self.std[i] = nonzero_data.std()
        else:
            self.mean = np.zeros(n_features)
            self.std = np.zeros(n_features)
            
            for i in range(n_features):
                feature_data = data[..., i]
                nonzero_mask = (feature_data != 0)
                if nonzero_mask.any():
                    nonzero_data = feature_data[nonzero_mask]
                    self.mean[i] = np.mean(nonzero_data)
                    self.std[i] = np.std(nonzero_data)
            
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """标准化变换，转换所有值"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
            
        result = data.clone() if isinstance(data, torch.Tensor) else data.copy()
        n_features = data.shape[-1]
        
        for i in range(n_features):
            if isinstance(data, torch.Tensor):
                result[..., i] = (data[..., i] - self.mean[i]) / (self.std[i] + self.eps)
            else:
                result[..., i] = (data[..., i] - self.mean[i]) / (self.std[i] + self.eps)
                    
        return result
            
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """逆变换，转换所有值"""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
            
        result = data.clone() if isinstance(data, torch.Tensor) else data.copy()
        n_features = data.shape[-1]
        
        for i in range(n_features):
            if isinstance(data, torch.Tensor):
                result[..., i] = data[..., i] * self.std[i] + self.mean[i]
            else:
                result[..., i] = data[..., i] * self.std[i] + self.mean[i]
                    
        return result       

class MinMaxScaler():
    """归一化: (x - min) / (max - min)"""
    def __init__(self, feature_range: Tuple[float, float] = (0, 1), eps: float = 1e-8):
        self.feature_range = feature_range
        self.eps = eps
        self.min = None
        self.max = None
        
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> None:
        """计算最小值和最大值"""
        if isinstance(data, torch.Tensor):
            self.min = data.min(0, keepdim=True)[0]
            self.max = data.max(0, keepdim=True)[0]
        else:
            self.min = np.min(data, axis=0, keepdims=True)
            self.max = np.max(data, axis=0, keepdims=True)
            
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """归一化变换"""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before transform")
            
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min + self.eps)
        min_scale = self.feature_range[0] - self.min * scale
            
        if isinstance(data, torch.Tensor):
            return data * scale + min_scale
        else:
            return data * scale + min_scale
            
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """逆变换"""
        if self.min is None or self.max is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
            
        scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min + self.eps)
        min_scale = self.feature_range[0] - self.min * scale
            
        if isinstance(data, torch.Tensor):
            return (data - min_scale) / scale
        else:
            return (data - min_scale) / scale

def point_mask(array, target_missing_rate=0.75, seed=None):
    """
    将张量中的缺失率增加到目标缺失率，并生成对应的 mask
    
    参数:
    tensor: 输入张量，其中 0 表示缺失值
    target_missing_rate: 目标缺失率，默认为 0.5
    seed: 随机种子，用于复现结果
    
    返回:
    new_tensor: 增加缺失值后的张量
    mask: 对应的 mask 张量，1 表示有值，0 表示缺失
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 创建数组的副本
    new_array = array.copy()
    
    # 计算当前缺失值的数量和比例
    total_elements = array.size
    current_missing = np.sum(array == 0)
    current_missing_rate = current_missing / total_elements
    
    if current_missing_rate >= target_missing_rate:
        print(f"当前缺失率 ({current_missing_rate:.2%}) 已经超过目标缺失率 ({target_missing_rate:.2%})")
        mask = (new_array != 0).astype(float)
        drop_mask = np.zeros_like(array, dtype=float)
        return new_array, mask, drop_mask
    
    # 计算需要额外丢弃的数量
    target_missing = int(total_elements * target_missing_rate)
    additional_missing = target_missing - current_missing
    
    # 创建当前非零值的 mask
    non_zero_mask = (array != 0)
    non_zero_count = np.sum(non_zero_mask)
    
    if non_zero_count < additional_missing:
        raise ValueError("非零元素不足以达到目标缺失率")
    
    # 创建随机丢弃的 mask
    # 在非零值中随机选择指定数量的位置
    drop_probs = np.random.rand(*array.shape)
    # 将已经是 0 的位置的概率设为无穷大（确保不会被选中）
    drop_probs[~non_zero_mask] = np.inf
    # 选择最小的 additional_missing 个概率值对应的位置
    drop_threshold = np.partition(drop_probs.flatten(), additional_missing)[additional_missing]
    drop_mask = (drop_probs <= drop_threshold).astype(float)
    
    # 应用 drop_mask
    new_array[drop_mask == 1] = 0
    
    # 生成最终的 mask
    mask = (new_array != 0).astype(float)
    
    # 验证最终缺失率
    final_missing_rate = np.sum(new_array == 0) / total_elements
    # print(f"原始缺失率: {current_missing_rate:.2%}")
    # print(f"最终缺失率: {final_missing_rate:.2%}")
    
    return new_array, mask, drop_mask

def node_mask(array, missing_ratio):
    """
    在节点维度上创建缺失数据，drop_mask只标记原本非0但新设为0的值
    
    参数:
    array: shape为[seq_len, node_size]的numpy数组
    missing_ratio: 需要完全缺失的节点比例
    
    返回:
    new_array: 添加了新缺失值的数组
    mask: 标记所有缺失值(包括原有和新增)的布尔掩码
    drop_mask: 仅标记新增缺失值的布尔掩码(原本为非0但被设为0的值)
    """
    seq_len, node_size, _ = array.shape
    
    # 计算需要完全缺失的节点数量
    num_missing_nodes = int(node_size * missing_ratio)
    
    # 随机选择要缺失的节点索引
    missing_nodes = np.random.choice(
        node_size, 
        size=num_missing_nodes, 
        replace=False
    )
    
    # 创建新的数组副本
    new_array = array.copy()
    
    # 创建drop_mask，初始化为False
    drop_mask = np.zeros_like(array, dtype=bool)
    
    # 对于每个选中的节点
    for node_idx in missing_nodes:
        # 找出该节点上原本非0的位置
        non_zero_mask = array[:, node_idx] != 0
        # 在drop_mask中标记这些位置
        drop_mask[:, node_idx] = non_zero_mask
        # 将该节点所有值设为0
        new_array[:, node_idx] = 0
    
    # 创建整体的mask（False表示缺失）
    mask = (new_array != 0)
    
    return new_array, mask, drop_mask