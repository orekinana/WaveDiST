from configs import BaseConfig

from .datamodules import SpatialTemporalImputationDataModule


def get_datamodule(configs: BaseConfig):
    return SpatialTemporalImputationDataModule(
        dataset_name=configs.data.dataset_name,
        batch_size=configs.data.batch_size,
        num_workers=configs.data.num_workers,
        seq_len=configs.data.seq_len,
        scale=configs.data.scale,
        mask_mode=configs.data.mask_mode,
        mask_ratio=configs.data.mask_ratio,
        test_mask_ratio=configs.data.test_mask_ratio,
        train_ratio=configs.data.train_ratio,
        val_ratio=configs.data.val_ratio,
    )
