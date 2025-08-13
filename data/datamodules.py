# data/datamodules/traffic_datamodule.py
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from .datasets import SpatialTemporalDataset
from .transforms import StandardScaler, MinMaxScaler

import os

class SpatialTemporalImputationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        seq_len: int = 12,
        scale: str = 'standard',
        mask_mode: str = 'node',
        mask_ratio: float = 0.5,
        test_mask_ratio: float = 0.5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        """
        Args:
            dataset_name: 数据集名称
            batch_size: 批次大小
            num_workers: 数据加载线程数
            seq_len: 输入序列长度
            scale: 是否进行数据标准化
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 数据集参数
        self.dataset_name = dataset_name
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.seq_len      = seq_len
        self.mask_mode    = mask_mode
        self.mask_ratio   = mask_ratio
        self.test_mask_ratio = test_mask_ratio
        
        # 数据划分比例
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        
        # 数据预处理
        self.scale = scale
        
        # 数据集对象
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None
    
    def setup(self, stage: Optional[str] = None):
        """数据集设置,在每个GPU上执行"""
        if stage == 'fit' or stage is None:
            
            data, adj = get_data(self.dataset_name)
            
            # 数据集划分
            train_end = int(len(data) * self.train_ratio)
            val_end = int(len(data) * (self.train_ratio + self.val_ratio))
            
            train_data = data[:train_end]
            # val_data = data[train_end:val_end]
            # test_data = data[val_end:]

            val_data = data[train_end:]
            test_data = data[train_end:]


            # 数据标准化
            if self.scale:
                if self.scale == 'standard':
                    self.scaler = StandardScaler()
                elif self.scale == 'minmax':
                    self.scaler = MinMaxScaler(feature_range=(0, 1))
                else:
                    raise ValueError(f"Unknown scale method: {self.scaler}")
                    
                # 只用训练集的统计量进行标准化
                self.scaler.fit(train_data)
                train_data = self.scaler.transform(train_data)
                val_data = self.scaler.transform(val_data)
                test_data = self.scaler.transform(test_data)
            
            # 创建数据集对象
            self.train_dataset = SpatialTemporalDataset(
                data=train_data,
                adj=adj,
                seq_len=self.seq_len,
                mask_mode=self.mask_mode,
                mask_ratio=self.mask_ratio
            )
            
            self.val_dataset = SpatialTemporalDataset(
                data=val_data,
                adj=adj,
                seq_len=self.seq_len,
                mask_mode=self.mask_mode,
                mask_ratio=self.mask_ratio
            )
            
            self.test_dataset = SpatialTemporalDataset(
                data=test_data,
                adj=adj,
                seq_len=self.seq_len,
                mask_mode=self.mask_mode,
                mask_ratio=self.test_mask_ratio
            )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

def get_data(dataset_name):
    if dataset_name == 'JN_taxi':
        return JN_taxi_data()
    elif dataset_name == 'XA_purchase':
        return purchase_data('xian')
    elif dataset_name == 'BJ_purchase':
        return purchase_data('beijing')
    elif dataset_name == 'JH_purchase':
        return purchase_data('jinhua')
    elif dataset_name == 'BJ_air':
        return air_data()
    elif dataset_name.startswith("PEMS"):
        return pems_data(dataset_name)
    elif dataset_name.startswith("METRLA"):
        return metrla_data()
    else:
        print('----------- The dataset is not exist! -----------')
    
def JN_taxi_data():
    # 加载 JN taxi 数据的邻接矩阵
    adj = np.load('/data/datasets/traffic/JN_didi/JN_adj.npy')

    # 存在已经处理好的 npy 文件则直接加载
    if os.path.exists('data/datacaches/traffic/data.npy'):
        speed_matrix = np.load('data/datacaches/traffic/data.npy')
        return speed_matrix, adj
    
    # 处理csv变成可训练的npy文件
    df = pd.read_csv("/data/datasets/traffic/JN_didi/JN_speed.csv")
    # 获取唯一的时间和节点，并建立映射
    times = sorted(df['datetime'].unique())
    nodes = sorted(df['nodeid'].unique())
    
    time_to_idx = {t: i for i, t in enumerate(times)}
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # 创建空矩阵
    speed_matrix = np.full((len(times), len(nodes)), np.nan)
    
    # 填充数据
    for _, row in df.iterrows():
        i = time_to_idx[row['datetime']]
        j = node_to_idx[row['nodeid']]
        speed_matrix[i, j] = row['speed']
    speed_matrix = np.expand_dims(speed_matrix, axis=-1)

    # 存储处理好的 npy 文件
    np.save('data/datacaches/traffic/data.npy', speed_matrix)

    return speed_matrix, adj



def purchase_data(city):
    data = np.load(f'/data/datasets/purchase/{city}/{city}.npy')
    data = np.expand_dims(data, axis=-1)
    adj = np.load(f'/data/datasets/purchase/{city}/{city}_adj.npy')
    return data, adj

def air_data():
    data = np.load(f'/data/datasets/air/air.npy')
    data = np.expand_dims(data, axis=-1)
    adj = np.load(f'/data/datasets/air/air_adj.npy')
    return data, adj

def pems_data(data_name):
    data = np.load(f'/data/datasets/traffic/PEMS/{data_name}/{data_name}.npz')['data'][:,:,0]
    data = np.expand_dims(data, axis=-1)
    adj = np.load(f'/data/datasets/traffic/PEMS/{data_name}/{data_name}_adj.npy')
    return data, adj

def metrla_data():
    data = np.load(f'/data/datasets/traffic/METR-LA/metr_la.npy')
    data = np.expand_dims(data, axis=-1)
    adj = np.load(f'/data/datasets/traffic/METR-LA/metr_la_adj.npy')
    return data, adj
