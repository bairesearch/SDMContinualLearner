# cd into the py_scripts directory before running this. 

import shutil 
import os
import CIFAR10_split_datasets
import CIFAR100_split_datasets
import MNIST_split_datasets
import CIFAR10_torchify
import CIFAR100_torchify
import ImageNet32_torchify
import CIFAR10_cached_data_split
import CIFAR10_cached_data_split_randomize
import compute_context_vector_MNIST 
import processing_MNIST 
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet

datasetMNIST = True
datasetCIFAR10 = True
datasetCIFAR100 = True
datasetImagenet = True

torchvisionDatasetDownloadList = []
if(datasetMNIST):
    torchvisionDatasetDownloadList.append(MNIST)
if(datasetCIFAR10):
    torchvisionDatasetDownloadList.append(CIFAR10)
if(datasetCIFAR100):
    torchvisionDatasetDownloadList.append(CIFAR100)
if(datasetImagenet):
    print("datasetImagenet: assume imagenet32 has been downloaded to ../data/ImageNet32/train/train_data_batch_x / test/test_data")

if __name__ == '__main__':

    data_path = '../data/'

    # get the datasets.
    for data_func in torchvisionDatasetDownloadList:
        # getting train and then test data!
        data_func(data_path, train=True, download=True)
        data_func(data_path, train=False, download=True)
        
    if(datasetMNIST):
        # process the MNIST data
        processing_MNIST.process_MNIST()

    if(datasetCIFAR10):
        # torchifying the CIFAR dataset too (leads to double the loading speed!)
        CIFAR10_torchify.make_dataset()
    if(datasetCIFAR100):
        CIFAR100_torchify.make_dataset()
    if(datasetImagenet):
        ImageNet32_torchify.make_dataset()
 
    split_dir = f'{data_path}splits/'
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)

    if(datasetCIFAR10):
        # moving the ConvMixerEmbeddings to the right folder. 
        shutil.copytree('../ConvMixerEmbeddings/', '../data/ConvMixerEmbeddings')

        # Splitting the ConvMixer embedded CIFAR10 data
        CIFAR10_cached_data_split.split_dataset()
        # making the other random seed splits too
        CIFAR10_cached_data_split_randomize.split_dataset_randomize_cached()
        CIFAR10_cached_data_split_randomize.split_dataset_randomize()
    
    # This makes the raw data splits
    if(datasetMNIST):
        if not os.path.isdir(split_dir+'MNIST'):
            os.mkdir(split_dir+'MNIST')
    if(datasetCIFAR10):
        if not os.path.isdir(split_dir+'CIFAR10'):
            os.mkdir(split_dir+'CIFAR10')
    if(datasetCIFAR100):
        if not os.path.isdir(split_dir+'CIFAR100'):
            os.mkdir(split_dir+'CIFAR100')
    if(datasetMNIST):
        MNIST_split_datasets.split_dataset()
    if(datasetCIFAR10):
        CIFAR10_split_datasets.split_dataset()
    if(datasetCIFAR100):
        CIFAR100_split_datasets.split_dataset()

    if(datasetMNIST):
        # making the context vectors for Active Dendrites: 
        compute_context_vector_MNIST.make_context_vectors()


