
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from configs import BaseConfig


def get_model_checkpoint(configs: BaseConfig) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=configs.log.checkpoint_dir,
        filename=f'{configs.data.dataset_name}-seqlen{configs.data.seq_len}-{configs.data.mask_mode}mask-maskratio{configs.data.mask_ratio}',
        monitor=configs.log.checkpoint_monitor,
        save_top_k=configs.log.checkpoint_save_top_k,
        every_n_epochs=configs.log.checkpoint_every_n_epochs,
        save_last=configs.log.checkpoint_save_last,
        save_on_train_epoch_end=configs.log.checkpoint_save_on_train_epoch_end,
        enable_version_counter=configs.log.enable_version_counter,
    )


def get_early_stop(configs: BaseConfig) -> EarlyStopping:
    return EarlyStopping(
        monitor=configs.train.early_stopping_monitor,
        patience=configs.train.early_stopping_patience,
        mode=configs.train.early_stopping_mode
    )


def setup_callbacks(configs: BaseConfig) -> list:
    """设置训练回调函数"""
    callbacks = []

    # 设置模型检查点保存
    if configs.log.save_checkpoint:
        callbacks.append(get_model_checkpoint(configs))

    # 设置早停
    if configs.train.early_stopping:
        callbacks.append(get_early_stop(configs))

    return callbacks
