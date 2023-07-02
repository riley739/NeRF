from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from my_method.pipeline import MyPipeline
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

MyMethod = MethodSpecification(
    config=TrainerConfig(
	    method_name="my-method",
	    steps_per_eval_batch=500,
	    steps_per_save=2000,
	    max_num_iterations=30000,
	    mixed_precision=True,
	    pipeline=MyPipeline(
	        datamanager=VanillaDataManagerConfig(
	            dataparser=NerfstudioDataParserConfig(),
	            train_num_rays_per_batch=1024
            ),
	        model=VanillaModelConfig(
	            _target=MipNerfModel,
	            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
	            num_coarse_samples=128,
	            num_importance_samples=128,
	            eval_num_rays_per_chunk=1024,
	        ),
	    ),
	    optimizers={
	        "fields": {
	            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
	            "scheduler": None,
	        }
	    },
	    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
	),
	description="My basic model!!!",
)