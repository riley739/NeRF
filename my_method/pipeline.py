import typing
from dataclasses import dataclass, field
from typing import Literal, Type

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)




class MyPipeline(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    