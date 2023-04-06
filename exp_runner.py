
import train_model
import importlib

#	'ablate_gaba_switch_timing',

expFiles = [
	'ablate_gaba_switch_timing',
	'ablate_output_layer_bias',
	'ablate_pretrain_k_vals_10K_Neurons_ImgNet32_embeddings',
	'ablate_pretrain_k_vals_1K_Neurons_ImgNet32_embeddings',
	'ablate_test_k_vals_10K_Neurons_SplitCIFAR10_embeddings',
	'ablate_test_k_vals_1K_Neurons_SplitCIFAR10_embeddings',
	'ablate_test_sdm_optimizers',
	'cont_learn_ConvSDM_SplitCIFAR10_pixels',
	'cont_learn_on_CIFAR100_embeddings',
	'cont_learn_on_SplitCIFAR10_embeddings_no_pretrain',
	'cont_learn_on_SplitCIFAR10_embeddings',
	'cont_learn_on_SplitCIFAR10_pixels',
	'cont_learn_on_SplitMNIST',
	'get_nice_receptive_fields_10K',
	'investigate_and_log_CIFAR10_stale_gradients',
	'investigate_cont_learning',
	'investigate_deadneuron_GABAswitch_vs_subtract',
	'oracle_train',
	'pretrain_ConvSDM_10K_Neurons_on_CIFAR10_pixels',
	'pretrain_ConvSDM_1K_Neurons_on_CIFAR10_pixels',
	'pretrain_ConvSDM_on_ImageNet_pixels',
	'pretrain_on_ImgNet32_embeddings',
	'pretrain_on_ImgNet32_pixels'
]

expFilesHyperParameterSweep = [
	'hyperparam_sweep_Dropout',
	'hyperparam_sweep_EWC',
	'hyperparam_sweep_L2',
	'hyperparam_sweep_MAS',
	'hyperparam_sweep_SI'
]


#expCommandsFolder = 'exp_commands'
expCommandsFolder = 'exp_commands_trialLowEpochs'

gbl = globals()
def force_import(module_name):
	try:
		gbl[module_name] = importlib.import_module(module_name)
	except ImportError:
		print("module_name ", module_name, " does not exist")
		exit()

for expFileIndex, expFile in enumerate(expFiles):
	print("expFileIndex = ", expFileIndex, ", expFile = ", expFile)
	module_name = expCommandsFolder + '.' + expFile	#. instead of / used for pythonic subfolder referencing
	force_import(module_name)
	for expIndex, expParameters in enumerate(gbl[module_name].exp_list):
		print("expIndex = ", expIndex)
		trainParameters = gbl[module_name].settings_for_all | expParameters
		model_style = trainParameters.pop('model_style')
		dataset = trainParameters.pop('dataset')
		load_path = None
		train_model.trainModel(model_style, dataset, trainParameters, load_path) 
	del gbl[module_name]
	
