from torch.utils.data import Dataset
import torch
import numpy as np
from .transforms import point_mask, node_mask

class SpatialTemporalDataset(Dataset):

    def __init__(
        self,
        data: np.ndarray,
        adj: np.ndarray,
        seq_len: int,
        mask_mode='node',
        mask_ratio=0.5
    ):
        """
        Args:
            data: 原始数据数组 [num_samples, num_nodes]
            seq_len: 输入序列长度
            transform: 数据转换器
        """
        super().__init__()
        self.data = data
        self.adj = adj
        self.seq_len = seq_len
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        
        # 生成样本索引
        self.indices = self._generate_indices()

        # mask输入数据
        if self.mask_mode == 'node':
            self.mask_data, self.mask, self.val_mask = node_mask(self.data, self.mask_ratio)
        else:
            self.mask_data, self.mask, self.val_mask = point_mask(self.data, self.mask_ratio)
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取当前样本的起始索引
        start_idx = self.indices[idx]

        # 切片获取输入序列和目标序列
        x_seq  = self.mask_data[start_idx:start_idx + self.seq_len]
        y_seq  = self.data[start_idx:start_idx + self.seq_len]
        m_seq  = self.mask[start_idx:start_idx + self.seq_len]
        vm_seq = self.val_mask[start_idx:start_idx + self.seq_len]
        
        # 转换为tensor
        x_seq  = torch.FloatTensor(x_seq)
        y_seq  = torch.FloatTensor(y_seq)
        m_seq  = torch.FloatTensor(m_seq)
        vm_seq = torch.FloatTensor(vm_seq)
        adj    = torch.FloatTensor(self.adj)
        
        
        sample = {
            'x_seq' : x_seq,  # [seq_len, spatial_dim]
            'y_seq' : y_seq,  # [seq_len, spatial_dim]
            'm_seq' : m_seq,  # [seq_len, spatial_dim]
            'vm_seq': vm_seq, # [seq_len, spatial_dim]
            'adj'   : adj     # [spatial_dim, spatial_dim]
        }
            
        return sample
    
    def _generate_indices(self):
        """生成有效的样本起始索引"""
        valid_indices = []
        total_len = len(self.data)
        
        for i in range(total_len - self.seq_len + 1):
            # 检查当前窗口是否有效
            if self._is_valid_window(i):
                valid_indices.append(i)
                
        return valid_indices
    
    def _is_valid_window(self, start_idx):
        """检查给定起始索引的窗口是否有效"""
        window = self.data[start_idx:
                          start_idx + self.seq_len]
        
        # 检查是否存在缺失值
        if np.isnan(window).any():
            return False
            
        # 其他有效性检查...
        return True
