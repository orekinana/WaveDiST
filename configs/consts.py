from typing import Literal


DATASET_NAME_TRAFFIC = 'traffic'
DATASET_NAME_SALE    = 'sale'
DATASET_NAME_AIR     = 'air'

MODEL_NAME_STFD                  = 'stfd'
MODEL_NAME_WO_DIFFUSION          = 'stfd_wo_diffusion'
MODEL_NAME_WO_FREQ               = 'stfd_wo_freq'
MODEL_NAME_IMPUTEFORMER          = 'imputeformer'
STFD_ATTENTION_TYPE_SEPARATE     = 'separate'
STFD_ATTENTION_TYPE_UNIFIED      = 'unified'
STFD_ATTENTION_TYPE_TEMPORAL     = 'temporal'
STFD_MODE_ENCODER_DECODER        = 'encoder_decoder'
STFD_MODE_DECODER_ONLY           = 'decoder_only'
STFD_TEMP_POS_EBD_SIN            = 'sinusoidal'
STFD_TEMP_POS_EBD_LEARNABLE      = 'learnable'
STFD_NODE_POS_EBD_LAPLACIAN      = 'laplacian'
STFD_SPECIAL_TOKENS_INIT_ZEROS   = 'zeros'
STFD_SPECIAL_TOKENS_INIT_RANDOM  = 'random'
STFD_SPECIAL_TOKENS_INIT_ONEHOT  = 'onehot'

# 自定义类型
STFDAttentionType     = Literal['separate', 'unified', 'temporal']
STFDMode              = Literal['encoder_decoder', 'decoder_only']
STFDTempPosEmbedding  = Literal['sinusoidal', 'learnable']
STFDNodePosEmbedding  = Literal['laplacian']
STFDSpecialTokensInit = Literal['zeros', 'random', 'onehot']
