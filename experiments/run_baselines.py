import pytorch_lightning as pl
from configs import BaseConfig
from data import get_datamodule

from pypots.imputation import Transformer, ImputeFormer, TimeMixer, TimesNet, FreTS, StemGNN, SAITS, ModernTCN, CSDI, GPVAE
from pypots.utils.metrics import calc_mae, calc_mse, calc_rmse
import numpy as np
import torch

def main():

    # 加载配置
    configs = BaseConfig.from_args()

    # 初始化数据模块
    datamodule = get_datamodule(configs)
    datamodule.setup()

    train_dataset = reshape_data(datamodule.train_dataset.mask_data.squeeze(axis=-1))
    train_dataset = np.where(train_dataset == 0, np.nan, train_dataset)

    val_dataset = reshape_data(datamodule.val_dataset.mask_data.squeeze(axis=-1))
    val_dataset = np.where(val_dataset == 0, np.nan, val_dataset)

    test_dataset = reshape_data(datamodule.test_dataset.mask_data.squeeze(axis=-1))
    test_dataset = np.where(test_dataset == 0, np.nan, test_dataset)

    val_mask = reshape_data(datamodule.test_dataset.val_mask.squeeze(axis=-1).astype(int))
    y = reshape_data(datamodule.test_dataset.data.squeeze(axis=-1))
    models = [
        # ('SAITS', SAITS(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.SAITS['n_layers'], d_model=configs.model.SAITS['d_model'], n_heads=configs.model.SAITS['n_heads'], d_k=configs.model.SAITS['d_k'], d_v=configs.model.SAITS['d_v'], d_ffn=configs.model.SAITS['d_ffn'], batch_size=configs.data.batch_size, epochs=configs.model.epochs)),
        # ('Transformer', Transformer(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.Transformer['n_layers'], d_model=configs.model.Transformer['d_model'], n_heads=configs.model.Transformer['n_heads'], d_k=configs.model.Transformer['d_k'], d_v=configs.model.Transformer['d_v'], d_ffn=configs.model.Transformer['d_ffn'], batch_size=configs.data.batch_size, epochs=configs.model.epochs)),
        # ('TimeMixer', TimeMixer(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.TimeMixer['n_layers'], d_model=configs.model.TimeMixer['d_model'], d_ffn=configs.model.TimeMixer['d_ffn'], top_k=configs.model.TimeMixer['top_k'], batch_size=configs.data.batch_size, epochs=configs.model.epochs)),
        # ('TimesNet', TimesNet(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.TimesNet['n_layers'], top_k=configs.model.TimesNet['top_k'], d_model=configs.model.TimesNet['d_model'], d_ffn=configs.model.TimesNet['d_ffn'],  n_kernels=configs.model.TimesNet['n_kernels'], batch_size=configs.data.batch_size, epochs=configs.model.epochs)),
        # ('FreTS', FreTS(n_steps=configs.data.seq_len, n_features=configs.model.n_features, batch_size=configs.data.batch_size, epochs=configs.model.epochs)),
        # ('ImputeFormer', ImputeFormer(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.ImputeFormer['n_layers'], d_input_embed=configs.model.ImputeFormer['d_input_embed'], d_learnable_embed=configs.model.ImputeFormer['d_learnable_embed'], d_proj=configs.model.ImputeFormer['d_proj'], d_ffn=configs.model.SAITS['d_ffn'], n_temporal_heads=configs.model.ImputeFormer['n_temporal_heads'], batch_size=configs.data.batch_size, epochs=configs.model.epochs)),
        # ('moderntcn', ModernTCN(n_steps=configs.data.seq_len, n_features=configs.model.n_features, patch_size=configs.model.ModernTCN['patch_size'], patch_stride=configs.model.ModernTCN['patch_stride'], downsampling_ratio=configs.model.ModernTCN['downsampling_ratio'], ffn_ratio=configs.model.ModernTCN['ffn_ratio'], num_blocks=configs.model.ModernTCN['num_blocks'], large_size=configs.model.ModernTCN['large_size'], small_size=configs.model.ModernTCN['small_size'], dims=configs.model.ModernTCN['dims'], epochs=configs.model.epochs)),
        ('csdi', CSDI(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.CSDI['n_layers'], n_heads=configs.model.CSDI['n_heads'], n_channels=configs.model.CSDI['n_channels'], d_time_embedding=configs.model.CSDI['d_time_embedding'], d_feature_embedding=configs.model.CSDI['d_feature_embedding'], d_diffusion_embedding=configs.model.CSDI['d_diffusion_embedding'], epochs=configs.model.epochs)),
        # ('stemgnn', StemGNN(n_steps=configs.data.seq_len, n_features=configs.model.n_features, n_layers=configs.model.TimeMixer['n_layers'], n_stacks=configs.model.StemGNN['n_stacks'], d_model=configs.model.StemGNN['d_model'], epochs=configs.model.epochs)),
        # ('gpvae', GPVAE(n_steps=configs.data.seq_len, n_features=configs.model.n_features, latent_size=configs.model.GPVAE['latent_size'], epochs=configs.model.epochs))
    ]

    for name, model in models:
        if configs.only_test:
            model.load(f'{configs.log.checkpoint_dir}/{name}-{configs.data.dataset_name}.pypots')
        else:
            if configs.load_model:
                model.load(f'{configs.log.checkpoint_dir}/{name}-{configs.data.dataset_name}.pypots')
            model.fit({"X": train_dataset})  # train the model on the dataset
            model.save(f'{configs.log.checkpoint_dir}/{name}-{configs.data.dataset_name}', overwrite=True)

        import time
        start = time.time()
        y_hat = model.impute({"X": test_dataset})  # impute the originally-missing values and artificially-missing values
        print(f"{name }【time】耗时: {time.time() - start:.4f}秒")

        orin_y_hat = datamodule.scaler.inverse_transform(np.expand_dims(y_hat, axis=-1))
        orin_y     = datamodule.scaler.inverse_transform(np.expand_dims(y, axis=-1))

        # orin_y     = orin_y.squeeze(axis=-1)
        # orin_y_hat = orin_y_hat.squeeze(axis=-1)

        orin_y     = orin_y.squeeze()
        orin_y_hat = orin_y_hat.squeeze()

        mae = calc_mae(orin_y_hat, orin_y, val_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
        mse = calc_mse(orin_y_hat, orin_y, val_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)
        rmse = calc_rmse(orin_y_hat, orin_y, val_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)

        print(f'{name} - ', 'mae: ', mae, 'mse: ', mse, 'rmse: ', rmse)

def reshape_data(data, seqlen=32):
    sample_size, _ = data.shape
    
    # 创建滑动窗口
    final_data = np.array([
        data[i:i + seqlen] 
        for i in range(0, sample_size - seqlen + 1, 1)
    ])  
    
    return final_data

def inv_mask_val(self, y_hat, y, val_mask):
    orin_y_hat = self.trainer.datamodule.scaler.inverse_transform(y_hat.cpu().detach().numpy())
    orin_y = self.trainer.datamodule.scaler.inverse_transform(y.cpu().detach().numpy())
    mask_orin_y_hat = torch.masked_select(torch.tensor(orin_y_hat).to(y_hat.device), val_mask.to(torch.bool))
    mask_orin_y     = torch.masked_select(torch.tensor(orin_y).to(y.device), val_mask.to(torch.bool))
    return mask_orin_y_hat, mask_orin_y

if __name__ == '__main__':
    main()
