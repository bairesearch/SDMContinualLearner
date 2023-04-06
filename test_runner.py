# simple test script to make sure that everything is working or easy debugging: 

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
import train_model

import wandb
from py_scripts.dataset_params import *
from py_scripts.combine_params import *

model_style = ModelStyles.SDM #ACTIVE_DENDRITES #FFN_TOP_K #CLASSIC_FFN #SDM
#dataset = DataSet.MNIST
#dataset = DataSet.CIFAR10
#dataset = DataSet.CIFAR100
#dataset = DataSet.ImageNet32
dataset = DataSet.SPLIT_MNIST
#dataset = DataSet.SPLIT_CIFAR10
#dataset = DataSet.SPLIT_CIFAR100
###
#dataset = DataSet.Cached_ConvMixer_CIFAR10
#dataset = DataSet.Cached_ConvMixer_ImageNet32
#dataset = DataSet.Cached_ConvMixer_CIFAR100	#does not work
###
#dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10
#dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_3
#dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_15
#dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_27
#dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR10_RandSeed_97
#dataset = DataSet.SPLIT_Cached_ConvMixer_ImageNet32	#does not work
###
#dataset = DataSet.SPLIT_CIFAR10_RandSeed_3
#dataset = DataSet.SPLIT_CIFAR10_RandSeed_15
#dataset = DataSet.SPLIT_CIFAR10_RandSeed_27
#dataset = DataSet.SPLIT_CIFAR10_RandSeed_97
#dataset = DataSet.SPLIT_Cached_ConvMixer_CIFAR100	#does not work
		
load_path = None

trainParameters = dict(
    num_workers=0, 
    epochs_to_train_for = 25,
    epochs_per_dataset = 5,
    cl_baseline = "EWC-MEMORY",
    #normalize_n_transform_inputs = True, 
    ewc_memory_beta=0.005,
    ewc_memory_importance=20000,
    #k_min=1, 
    #num_binary_activations_for_gaba_switch = 100000,
    #cl_baseline="MAS", # 'MAS', 'EWC-MEMORY', 'SI', 'L2', '
    #dropout_prob = 0.1,
    adversarial_attacks=False,
)
		
train_model.trainModel(model_style, dataset, trainParameters, load_path=None) 
