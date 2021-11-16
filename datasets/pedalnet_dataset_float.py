# Test dataset for audio filtering
# input and output both audio files

import numpy as np
import torch
from torch.utils.data import TensorDataset

import pickle

sample_size = 4410 #HAS to be the same as x_train.size(2)
out_sample_size = 2362

# class normalize:
#     def __init__(self, args):
#         self.args = args

#     def __call__(self, img):
#         if self.args.act_mode_8bit:
#             return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
#         return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127).div(128.)

def pedalnet_get_datasets(data, load_train=True, load_test=True):
    """
    data is a tuple of the specified data directory and the program arguments,
    and the two bools specify whether training and/or test data should be loaded.
    
    The loader returns a tuple of two PyTorch Datasets for training and test data.
    """

    (data_dir, args) = data

    #read data from pickle
    ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    data = pickle.load(open(data_dir + "/AUDIO/data.pickle", "rb"))  

    #concatenate train and validation, as in train.py the splitting happens again
    x_train = np.concatenate((data["x_train"],data["x_valid"]))
    y_train = np.concatenate((data["y_train"],data["y_valid"]))

    x_test = data["x_test"]
    y_test = data["y_test"]

    #transform data between -1 and 1 or -128 and 127
    x_train = x_train/x_train.max()
    y_train = y_train/y_train.max()

    x_test = x_test/x_test.max()
    y_test = y_test/y_test.max()
    
    if args.act_mode_8bit:
        x_train = (x_train*0.5*256.).round().clip(min=-128, max=127)
        y_train = (y_train*0.5*256.).round().clip(min=-128, max=127)

        x_test = (x_test*0.5*256.).round().clip(min=-128, max=127)
        y_test = (y_test*0.5*256.).round().clip(min=-128, max=127)
    else:
        #x_train = (x_train*0.5*256.).round().clip(min=-128, max=127)/(128.)
        #y_train = (y_train*0.5*256.).round().clip(min=-128, max=127)/(128.)

        #x_test = (x_test*0.5*256.).round().clip(min=-128, max=127)/(128.)
        #y_test = (y_test*0.5*256.).round().clip(min=-128, max=127)/(128.)

        pass


    #IDENTITY MAPPING EXPERIMENT!
    #y_train = x_train.copy()
    #y_test = x_test.copy()

    #sample_size = x_train.size(2)
    #EXPERIMENTAL: pre reduce y size (loss  = error_to_signal(y[:, :, -y_pred.size(2) :], y_pred).mean())
    pred_size = out_sample_size#sample_size-4
    y_train = y_train[:,:,-pred_size:]
    y_test = y_test[:,:,-pred_size:]

    train_ds = ds(x_train,y_train)
    test_ds = ds(x_test, y_test)
    
    #train_ds = ds(data["x_train"], data["y_train"])
    #valid_ds = ds(data["x_valid"], data["y_valid"])
    #test_ds = ds(data["x_test"], data["y_test"])


    return train_ds,test_ds

datasets = [
    {
        'name': 'PEDALNET_FLOAT',
        'input': (1, sample_size), #1 channel and 1D
        #'output': list(map(str, range(10))), labels: only for NOT regression
        'output': [1], #WHY do I need to put this shit here...
        'loader': pedalnet_get_datasets,
        'regression': True 
    },
]