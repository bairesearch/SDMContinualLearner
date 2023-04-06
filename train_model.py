# simple test script to make sure that everything is working or easy debugging: 

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from py_scripts.dataset_params import *
from py_scripts.combine_params import *


def trainModel(model_style, dataset, trainParameters, load_path=None):

	if load_path:
		print("LOADING IN A MODEL!!!")

	model_params, model, data_module = get_params_net_dataloader(model_style, dataset, load_from_checkpoint=load_path, **trainParameters)

	wandb_logger = None #WandbLogger(project="SDMContLearning", entity="YOURENTITY", save_dir="wandb_Logger/")
	model_params.logger = wandb_logger
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if torch.cuda.is_available():
		print("using cuda", device)
		gpu = [0]
	else: 
		print("using cpu", device)
		gpu=None

	# SETUP TRAINER
	if model_params.load_from_checkpoint and model_params.load_existing_optimizer_state:
		fit_load_state = load_path
	else: 
		fit_load_state = None

	callbacks = []
	# by default we will not save the model being trained unless it is in the continual learning setting. 
	if model_params.investigate_cont_learning: 
		num_checkpoints_to_keep = -1 
		model_checkpoint_obj = pl.callbacks.ModelCheckpoint(
			every_n_epochs = model_params.checkpoint_every_n_epochs,
			save_top_k = num_checkpoints_to_keep,
		)
		callbacks.append(model_checkpoint_obj)
		checkpoint_callback = True 
	else: 
		checkpoint_callback = False

	temp_trainer = pl.Trainer(
			logger=model_params.logger,
			max_epochs=model_params.epochs_to_train_for,
			check_val_every_n_epoch=1,
			num_sanity_val_steps = False,
			enable_progress_bar = True,
			accelerator="gpu", 
			callbacks = callbacks,
			reload_dataloaders_every_n_epochs=model_params.epochs_per_dataset, 

			)
	temp_trainer.fit(model, data_module)
	wandb.finish()
