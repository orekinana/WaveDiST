from pytorch_wavelets import DWT1D, IDWT1D
import pywt
import numpy as np
import torch


class WaveletNoiseTransformer:
    def __init__(self, wavelet='db1', level=None):
        """
        初始化小波变换器
        Args:
            wavelet: 小波基函数，例如'haar'/'db1', 'db4', 'morl'等
            level: 分解层级，如果为None则自动计算最大可分解层级
        """
        self.wavelet = wavelet
        print(f'wavelet base function is {wavelet}!')
        self.idwt = IDWT1D(wave=wavelet).cuda().half()
        
    def transform(self, data, level=3):

        data_numpy = np.array(data.cpu())

        """与之前实现相同的transform方法"""
        batch_size, window_size, node_size, feature_size = data.shape
        
        if level is None:
            level = pywt.dwt_max_level(window_size, pywt.Wavelet(self.wavelet).dec_len)
        
        batch_coeffs = [[] for _ in range(level + 1)]
        for batch_idx in range(batch_size):
            node_coeffs = [[] for _ in range(level + 1)]
            for node_idx in range(node_size):
                feature_coeffs = [[] for _ in range(level + 1)]
                for feature_idx in range(feature_size):
                    signal = data_numpy[batch_idx, :, node_idx, feature_idx]
                    coeffs = pywt.wavedec(signal, self.wavelet, level=level)
                    for i in range(level + 1):
                        feature_coeffs[i] = coeffs[i]
                for i in range(level + 1):
                    node_coeffs[i].append(feature_coeffs[i])
            for i in range(level + 1):
                batch_coeffs[i].append(node_coeffs[i])
        for i in range(level + 1):
            batch_coeffs[i] = torch.Tensor(batch_coeffs[i]).to(data.device)

        return batch_coeffs
    
    def add_noise_to_coeffs(self, coeffs, noise_schedule, noise_type='gaussian'):
        """
        对小波系数添加噪声
        Args:
            coeffs: 小波系数
            noise_schedule: 噪声调度，可以是：
                          - 字典：{level: noise_strength}
                          - 函数：level -> noise_strength
                          - 标量：所有层级使用相同的噪声强度
            noise_type: 噪声类型，'gaussian'或'uniform'
        Returns:
            noisy_coeffs: 加噪后的小波系数
        """
        # 深拷贝系数以避免修改原始数据
        noisy_coeffs = []
        
        # 标准化noise_schedule为字典形式
        if callable(noise_schedule):
            noise_dict = {i: noise_schedule(i) for i in range(self.level + 1)}
        elif isinstance(noise_schedule, (int, float)):
            noise_dict = {i: noise_schedule for i in range(self.level + 1)}
        else:
            noise_dict = noise_schedule
            
        # 对每个批次处理
        for batch_coeffs in coeffs:
            noisy_batch = []
            for node_coeffs in batch_coeffs:
                noisy_node = []
                for feature_coeffs in node_coeffs:
                    noisy_feature = []
                    
                    # 对每个层级的系数添加噪声
                    for level, coeff in enumerate(feature_coeffs):
                        noise_strength = noise_dict.get(level, 0.0)
                        
                        if noise_strength > 0:
                            # 计算该层级系数的标准差
                            coeff_std = np.std(coeff) if np.std(coeff) > 0 else 1.0
                            
                            # 生成噪声
                            if noise_type == 'gaussian':
                                noise = np.random.normal(0, coeff_std * noise_strength, coeff.shape)
                            else:  # uniform
                                noise = np.random.uniform(-coeff_std * noise_strength, 
                                                        coeff_std * noise_strength, 
                                                        coeff.shape)
                            
                            # 添加噪声
                            noisy_coeff = coeff + noise
                        else:
                            noisy_coeff = coeff.copy()
                            
                        noisy_feature.append(noisy_coeff)
                        
                    noisy_node.append(noisy_feature)
                noisy_batch.append(noisy_node)
            noisy_coeffs.append(noisy_batch)
            
        return noisy_coeffs
    
    def diffusion_like_noise(self, data, t, noise_schedule='linear'):
        """
        实现类似扩散模型的噪声添加过程
        Args:
            data: 原始数据
            t: 时间步 (0到1之间)
            noise_schedule: 噪声调度类型，'linear'或'cosine'
        Returns:
            noisy_data: 加噪后的数据
        """
        # 获取小波系数
        coeffs = self.transform(data)
        
        # 定义噪声强度调度
        if noise_schedule == 'linear':
            schedule_fn = lambda level: t * (level + 1) / (self.level + 1)
        else:  # cosine
            schedule_fn = lambda level: t * (1 - np.cos(np.pi * level / (self.level + 1))) / 2
            
        # 添加噪声
        noisy_coeffs = self.add_noise_to_coeffs(coeffs, schedule_fn)
        
        # 重建信号
        return self.reconstruct(noisy_coeffs)
    