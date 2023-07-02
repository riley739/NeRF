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
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from my_method.my_dataManager import MyDataManagerConfig, MyDataManager
from my_method.my_nerf import MyModelConfig, MyNeRFModel

@dataclass
class MyPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: MyPipeline)
    """target class to instantiate"""
    datamanager: MyDataManagerConfig = MyDataManagerConfig()
    """specifies the datamanager config"""
    model: MyModelConfig = MyModelConfig()
    """specifies the model config"""


class MyPipeline(VanillaPipeline):
    def __init__(
        self,
        config: MyPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        
        self.datamanager: MyDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(MyNeRFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
