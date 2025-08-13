import torch

from configs import BaseConfig, ModelConfig, MODEL_NAME_STFD, MODEL_NAME_WO_DIFFUSION, MODEL_NAME_WO_FREQ

from .networks.stfd import STDiffusionModel
from .networks.stfd_wo_diffusion import STWODiffusionModel
from .networks.stfd_wo_freq import STWOFreqModel

from .modules.impute import ImputationModule

def get_model(configs: BaseConfig) -> torch.nn.Module:
    if configs.model.name == MODEL_NAME_STFD:
        return STDiffusionModel(**_model_params(configs.model))
    if configs.model.name == MODEL_NAME_WO_DIFFUSION:
        return STWODiffusionModel(**_model_params(configs.model))
    if configs.model.name == MODEL_NAME_WO_FREQ:
        return STWOFreqModel(**_model_params(configs.model))
    else:
        return STDiffusionModel(**_model_params(configs.model))


def _model_params(configs: ModelConfig) -> dict:
    all_params = configs.model_dump()
    params = {}
    for k, v in all_params.items():
        if not k.startswith(configs.name[:4]):
            continue
        new_k = k[len(configs.name[:4])+ 1:]
        params[new_k] = v
    return params
