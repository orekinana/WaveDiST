import torch
import pytorch_lightning as pl
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
)

from configs import OptimizerConfig
class ImputationModule(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, optimizer_configs: OptimizerConfig):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.optimizer_configs = optimizer_configs

        # 统计指标
        metrics = MetricCollection({
            'mae': MeanAbsoluteError(),
            'mse': MeanSquaredError(),
            'rmse': MeanSquaredError(squared=False),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')


    def forward(self, x, adj):
        return self.model(x, adj)


    def training_step(self, batch):
        x, y, m, vm, adj = batch['x_seq'], batch['y_seq'], batch['m_seq'], batch['vm_seq'], batch['adj']
        import time
        start = time.time()
        y_hat = self(x, adj)
        print(f"【time】耗时: {time.time() - start:.4f}秒")

        # 计算和记录 Loss 和统计指标
        loss = self.model.loss_fn(y_hat, y, vm+m)
        mask_orin_y_hat, mask_orin_y = self.inv_mask_val(y_hat, y, vm)
        metrics = self.train_metrics(mask_orin_y_hat, mask_orin_y)
        self.log('train_loss', loss, prog_bar=True)
        self.log_dict(metrics, prog_bar=True)

        return loss


    def validation_step(self, batch):
        x, y, m, vm, adj = batch['x_seq'], batch['y_seq'], batch['m_seq'], batch['vm_seq'], batch['adj']
        y_hat = self(x, adj)

        # 计算并记录指标
        mask_orin_y_hat, mask_orin_y = self.inv_mask_val(y_hat, y, vm)
        metrics = self.val_metrics(mask_orin_y_hat, mask_orin_y)
        self.log_dict(metrics, prog_bar=True)


    def test_step(self, batch):
        x, y, m, vm, adj = batch['x_seq'], batch['y_seq'], batch['m_seq'], batch['vm_seq'], batch['adj']
        import time
        start = time.time()
        y_hat = self(x, adj)
        print(f"【time】耗时: {time.time() - start:.4f}秒")

        # 计算并记录指标
        mask_orin_y_hat, mask_orin_y = self.inv_mask_val(y_hat, y, vm)
        metrics = self.test_metrics(mask_orin_y_hat, mask_orin_y)
        self.log_dict(metrics, prog_bar=True)
        # self.save_result(y_hat, y, vm)
    
    def save_result(self, y_hat, y, val_mask):
        orin_y_hat      = self.trainer.datamodule.scaler.inverse_transform(y_hat.cpu().detach().numpy())
        orin_y          = self.trainer.datamodule.scaler.inverse_transform(y.cpu().detach().numpy())

        # 每次循环调用
        self.save_append_tensors({
            'orin': torch.tensor(orin_y),
            'pred': torch.tensor(orin_y_hat), 
            'mask': torch.tensor(val_mask)
        }, 'data/results/traffic_results.pt')

    def save_append_tensors(self, new_tensors, filepath):
        import os
        if not os.path.exists(filepath):
            torch.save(new_tensors, filepath)
            return new_tensors
            
        existing = torch.load(filepath)
        combined = {
            'orin': torch.cat([existing['orin'], new_tensors['orin']], dim=0),
            'pred': torch.cat([existing['pred'], new_tensors['pred']], dim=0), 
            'mask': torch.cat([existing['mask'], new_tensors['mask']], dim=0)
        }
        torch.save(combined, filepath)
        return combined

    def inv_mask_val(self, y_hat, y, val_mask):
        if val_mask.sum().item() == 0:
            return torch.Tensor([0]), torch.Tensor([0])
        orin_y_hat      = self.trainer.datamodule.scaler.inverse_transform(y_hat.cpu().detach().numpy())
        orin_y          = self.trainer.datamodule.scaler.inverse_transform(y.cpu().detach().numpy())
       
        mask_orin_y_hat = torch.masked_select(torch.tensor(orin_y_hat).to(y_hat.device), val_mask.to(torch.bool))
        mask_orin_y     = torch.masked_select(torch.tensor(orin_y).to(y.device), val_mask.to(torch.bool))

        return mask_orin_y_hat, mask_orin_y

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_configs.optimizer_type)(
            self.parameters(),
            **self.optimizer_configs.optimizer_params,
            weight_decay=self.optimizer_configs.weight_decay,
        )

        if not self.optimizer_configs.scheduler_enabled:
            return optimizer

        scheduler = getattr(torch.optim.lr_scheduler, self.optimizer_configs.scheduler_type)(
            optimizer,
            **self.optimizer_configs.scheduler_params
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self.optimizer_configs.scheduler_monitor,
                'interval': self.optimizer_configs.scheduler_interval,
                'frequency': self.optimizer_configs.scheduler_frequency
            }
        }
