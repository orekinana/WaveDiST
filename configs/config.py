from pydantic import BaseModel, Field
import yaml
import argparse
from typing import Any, Type, TypeVar

from .consts import *


T = TypeVar('T', bound='BaseConfig')


class DataConfig(BaseModel):
    '''Data configuration'''
    name:              str   = Field(default=DATASET_NAME_TRAFFIC, description='Dataset name')
    dataset_name:      str   = Field(default=None, description='Dataset name')
    batch_size:        int   = Field(default=32, description='Training batch size')
    num_workers:       int   = Field(default=4, description='Training batch size')
    seq_len:           int   = Field(default=12, description='Training batch size')
    scale:             str   = Field(default='standard', description='Training batch size')
    mask_mode:         str   = Field(default='node', description='Training batch size')
    mask_ratio:        float = Field(default=0.5, description='Training batch size')
    test_mask_ratio:   float = Field(default=0.5, description='Training batch size')
    train_ratio:       float = Field(default=0.7, description='Training batch size')
    val_ratio:         float = Field(default=0.2, description='Training batch size')


class ModelConfig(BaseModel):
    '''Model architecture configuration'''

    # 模型名称
    name:   str = Field(default='stfd', description='Name of model')
    epochs: int = Field(default=10, description='Training epochs')

    # STFD 模型相关配置
    stfd_dim_feature:            int                     = Field(default=None,                            description='')
    stfd_dim_seq:                int                     = Field(default=None,                            description='')
    stfd_dim_node:               int                     = Field(default=None,                            description='')
    stfd_num_heads:              int                     = Field(default=None,                            description='')
    stfd_num_encoder_layers:     int                     = Field(default=None,                            description='')
    stfd_num_decoder_layers:     int                     = Field(default=None,                            description='')
    stfd_max_seq_len:            int                     = Field(default=None,                            description='')
    stfd_dim_feedforward:        int                     = Field(default=2048,                            description='')
    stfd_dropout:                float                   = Field(default=0.1,                             description='')
    stfd_encoder_attention_type: 'STFDAttentionType'     = Field(default=STFD_ATTENTION_TYPE_SEPARATE,    description='')
    stfd_decoder_attention_type: 'STFDAttentionType'     = Field(default=STFD_ATTENTION_TYPE_SEPARATE,    description='')
    stfd_mode:                   'STFDMode'              = Field(default=STFD_MODE_ENCODER_DECODER,       description='Hidden layer size')
    stfd_temporal_pos_embedding: 'STFDTempPosEmbedding'  = Field(default=STFD_TEMP_POS_EBD_SIN,           description='Hidden layer size')
    stfd_node_pos_embedding:     'STFDNodePosEmbedding'  = Field(default=STFD_NODE_POS_EBD_LAPLACIAN,     description='Hidden layer size')
    stfd_special_tokens_init:    'STFDSpecialTokensInit' = Field(default=STFD_SPECIAL_TOKENS_INIT_RANDOM, description='Hidden layer size')
    noise_schedule:              str                     = Field(default='linear',                        description='')
    diffusion_steps:             int                     = Field(default=1000,                            description='')
    freq:                        bool                    = Field(default=True,                            description='')
    diffusion:                   bool                    = Field(default=True,                            description='')

    # Model general parameters
    n_features: int = Field(default=437, description='Number of features in the input data')

    # Baselines specific parameters
    SAITS:        dict[str, Any] = Field(default_factory=dict,        description='')
    ImputeFormer: dict[str, Any] = Field(default_factory=dict,        description='')
    Transformer:  dict[str, Any] = Field(default_factory=dict,        description='')
    TimeMixer:    dict[str, Any] = Field(default_factory=dict,        description='')
    TimesNet:     dict[str, Any] = Field(default_factory=dict,        description='')
    FreTS:        dict[str, Any] = Field(default_factory=dict,        description='')
    ModernTCN:    dict[str, Any] = Field(default_factory=dict,        description='')
    CSDI:         dict[str, Any] = Field(default_factory=dict,        description='')
    StemGNN:      dict[str, Any] = Field(default_factory=dict,        description='')
    GPVAE:        dict[str, Any] = Field(default_factory=dict,        description='')



class OptimizerConfig(BaseModel):
    '''Optimizer configuration'''
    optimizer_type:    str            = Field(default='Adam',              description='')
    optimizer_params:  dict[str, Any] = Field(default_factory=dict,        description='')
    weight_decay:      float          = Field(default=0.0,                 description='L2 regularization parameter')

    scheduler_enabled:   bool           = Field(default=False,               description='')
    scheduler_type:      str            = Field(default='CosineAnnealingLR', description='')
    scheduler_params:    dict[str, Any] = Field(default_factory=dict,        description='')
    scheduler_interval:  str            = Field(default='epoch',             description='')
    scheduler_monitor:   str            = Field(default='train_loss',          description='')
    scheduler_frequency: int            = Field(default=1,                   description='')


class TrainConfig(BaseModel):
    '''Training related configuration'''
    max_epochs: int   = Field(default=100, description='Number of max training epochs')

    # 加速相关
    accelerator:             str   = Field(default='auto', description='')
    strategy:                str   = Field(default=None, description='')
    precision:               int   = Field(default=32, description='')
    accumulate_grad_batches: int   = Field(default=1, description='')
    gradient_clip_val:       float = Field(default=None, description='')

    # early stopping 相关
    early_stopping:          bool  = Field(default=False, description='Whether early stop')
    early_stopping_monitor:  str   = Field(default='val_mse', description='')
    early_stopping_patience: int   = Field(default=10, description='')
    early_stopping_mode:     str   = Field(default='min', description='')


class TestConfig(BaseModel):
    '''Testing related configuration'''
    pass


class LogConfig(BaseModel):
    '''Logging related configuration'''
    log_dir:                            str  = Field(default='.', description='Logging dir')
    enable_progress_bar:                bool = Field(default=False, description='')
    log_every_n_steps:                  int  = Field(default=50, description='')
    save_checkpoint:                    bool = Field(default=True, description='Whether save checkpoint')
    enable_version_counter:             bool = Field(default=False, description='Whether append a version for checkpoint.')
    checkpoint_dir:                     str  = Field(default='.', description='')
    # checkpoint_filename:                str  = Field(default='{epoch}-{val_mse:.2f}', description='')
    checkpoint_monitor:                 str  = Field(default='val_mse', description='')
    checkpoint_every_n_epochs:          int  = Field(default=1, description='')
    checkpoint_save_top_k:              int  = Field(default=3, description='')
    checkpoint_save_last:               bool = Field(default=False, description='')
    checkpoint_save_on_train_epoch_end: bool = Field(default=False, description='')


class BaseConfig(BaseModel):
    '''Base configuration class'''

    # Sub-configurations
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    test: TestConfig = Field(default_factory=TestConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    # Basic configurations
    seed: int = Field(default=0, description='Random seed')
    device: str = Field(default='auto', description='Device to use')
    exp_name: str = Field(default=None, description='Name of experiment')
    exp_version: str = Field(default=None, description='Version of experiment')
    only_test: bool = Field(default=True, description='Whether do test')
    load_model: bool = Field(default=False, description='Whether do test')


    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        '''Load configuration from YAML file'''
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

    @classmethod
    def from_args(cls: Type[T]) -> T:
        '''Create configuration from command line arguments'''
        parser = argparse.ArgumentParser()

        # Add argument for yaml config path
        parser.add_argument('--config', type=str, help='Path to YAML config file')

        # Dynamically add arguments based on model fields
        for field_name, field in cls.model_fields.items():
            if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
                # Handle nested configs
                for sub_field_name, sub_field in field.annotation.model_fields.items():
                    parser.add_argument(
                        f'--{field_name}.{sub_field_name}',
                        type=sub_field.annotation,
                        help=sub_field.description,
                        default=None
                    )
            else:
                # Handle basic configs
                parser.add_argument(
                    f'--{field_name}',
                    type=field.annotation,
                    help=field.description,
                    default=None
                )

        args = parser.parse_args()

        # First load from yaml if specified
        config = cls() if not args.config else cls.from_yaml(args.config)

        # Update with command line arguments if specified
        args_dict = vars(args)
        for key, value in args_dict.items():
            if key != 'config' and value is not None:
                if '.' in key:
                    # Handle nested configs
                    main_key, sub_key = key.split('.')
                    getattr(config, main_key).__setattr__(sub_key, value)
                else:
                    config.__setattr__(key, value)

        return config
