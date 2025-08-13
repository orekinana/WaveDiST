import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import torch

from configs import BaseConfig
from models import (
    get_model,
    ImputationModule,
)
from data import get_datamodule
from utils import setup_callbacks


def main():
    # 加载配置
    configs = BaseConfig.from_args()

    # 设置随机种子
    pl.seed_everything(configs.seed)

    # 初始化数据模块
    datamodule = get_datamodule(configs)
    datamodule.setup()

    # 初始化模型
    model = get_model(configs)

    # 设置logger
    logger = TensorBoardLogger(
        save_dir=configs.log.log_dir,
        name=configs.exp_name,
        version=configs.exp_version,
    )

    # 设置callbacks
    callbacks = setup_callbacks(configs)

    # 设置训练器
    trainer = pl.Trainer(
        max_epochs=configs.train.max_epochs,
        accelerator=configs.train.accelerator,
        devices=configs.device,
        strategy=DDPStrategy() if configs.train.strategy == 'ddp' else 'auto',
        precision=configs.train.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=configs.train.gradient_clip_val,
        accumulate_grad_batches=configs.train.accumulate_grad_batches,
        log_every_n_steps=configs.log.log_every_n_steps,
        enable_progress_bar=configs.log.enable_progress_bar,
        enable_checkpointing=True,
    )

    if configs.only_test:
        model_path = configs.log.checkpoint_dir + '/' + callbacks[0].filename + '.ckpt'
        ckpt = torch.load(model_path)
        module = ImputationModule.load_from_checkpoint(model_path, model=model, **ckpt['hyper_parameters'])
    else:
        if configs.load_model:
            model_path = configs.log.checkpoint_dir + '/' + callbacks[0].filename + '.ckpt'
            ckpt = torch.load(model_path)
            module = ImputationModule.load_from_checkpoint(model_path, model=model, **ckpt['hyper_parameters'])
        else:
            module = ImputationModule(model, configs.optimizer)
        # 开始训练
        trainer.fit(module, datamodule=datamodule)

    # 测试模型
    # if configs.only_test:
    trainer.test(module, datamodule=datamodule)
    



if __name__ == '__main__':
    main()
