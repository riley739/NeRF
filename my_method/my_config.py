from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)

from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig

from my_method.my_dataManager import MyDataManagerConfig
from my_method.my_nerf import MyModelConfig
from my_method.my_pipeline import MyPipeline

MyMethod = MethodSpecification(
    config=TrainerConfig(
	    method_name="my-method",
	    steps_per_eval_batch=500,
	    steps_per_save=2000,
	    max_num_iterations=30000,
	    mixed_precision=True,
	    pipeline=MyPipeline(
	        datamanager=MyDataManagerConfig(
	            dataparser=NerfstudioDataParserConfig(),
	            train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=1024,
                camera_optimizer=CameraOptimizerConfig(
					mode="SO3xR3",
					optimizer=RAdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-3),
				),
            ),
			model=MyModelConfig(
            	eval_num_rays_per_chunk=1 << 15,
				num_nerf_samples_per_ray=128,
				num_proposal_samples_per_ray=(512, 256),
				hidden_dim=128,
				hidden_dim_color=128,
				appearance_embed_dim=128,
				base_res=32,
				max_res=4096,
				proposal_weights_anneal_max_num_iters=5000,
				log2_hashmap_size=21,
			),
		),
	    optimizers={
			"proposal_networks": {
				"optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
				"scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
			},
	        "fields": {
	            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
	            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
	        },
			"encodings": {
            	"optimizer": AdamOptimizerConfig(lr=0.02),
            	"scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
        	},
	    },
	    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
	),
	description="My basic model!!!",
)