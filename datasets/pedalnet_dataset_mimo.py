# Test dataset for audio filtering
# input and output both audio files

# MULTIPLE IN AND OUT CHANNELS

import numpy as np
import torch
from torch.utils.data import TensorDataset

import pickle

from scipy.signal import butter, lfilter

sampling_rate = 44100
sample_size = 4410 #HAS to be the same as x_train.size(2)
out_sample_size = 4410 #2362 # 2362 this is overridden by loss function, this has to be big enough! (smaller is better for performance)
#n_input_channels = 16

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    #high = highcut / nyq
    b, a = butter(order, [low], btype='highpass')
    return b, a

def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def create_dithering_noise(shape,sr,std,lowcut,order=1,pdf="normal"):
    """Creates dithering noises of the given shape, to be added to  signal
    Args:
        shape: shape of the array
        sr: sampling rate
        std: in normal case, std deviation with respect to quant level, in triangualr case is max noise value
        lowcut: cutoff freq of highpass filter on dithering noise
        order: order of highpass filter
        pdf: "triangular" or "normal"
    """
    # dithering noise (gaussian psd)
    #N = x_test_f.size #input (x lenght)
    q_step = (1./128)
    dither_std = std*q_step #lowest with still good results
    if pdf=="normal":
        dither_noise = np.random.randn(*shape)*dither_std
    elif pdf=="triangular":
        dither_noise = np.random.triangular(-dither_std,0,dither_std,shape)
    else:
        raise Exception("Invalid noise pdf",pdf) 
    #noise shaping / dither filtering
    #dither_noise_2 = butter_highpass_filter(dither_noise_2, lowcut, sr, order=order)
    dither_noise = butter_highpass_filter(dither_noise, lowcut, sr, order=order)
    return dither_noise.astype(np.float32)


def pedalnet_get_datasets(n_input_channels,n_output_channels,data, load_train=True, load_test=True):
    """
    data is a tuple of the specified data directory and the program arguments,
    and the two bools specify whether training and/or test data should be loaded.
    
    The loader returns a tuple of two PyTorch Datasets for training and test data.

    Usage: (shitty documentation style, you're welcome mdm)
    -Uses wide output (up to 32 bit). Can define output bit "amplitude" to actually use.
        args.output_bitdepth: 8 default, max 32
    -Dithering: adds normally sampled dithering
        --args.dither_std: std of dithering * quant level! (1= 1 quant level), default 0 (dither not present)
        --args.dither_hicutoff: cutoff freq of first order highpass filter on dither noise
    
    NOT IMPLEMENTED args.oversample: if 2 -> uses oversampled dataset (2x), default 1<
    """

    (data_dir, args) = data

    #read data from pickle
    ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

    data_file = "/AUDIO/data.pickle"
    
    data = pickle.load(open(data_dir + data_file, "rb"))  

    #concatenate train and validation, as in train.py the splitting happens again
    x_train = np.concatenate((data["x_train"],data["x_valid"]))
    y_train = np.concatenate((data["y_train"],data["y_valid"]))

    x_test = data["x_test"]
    y_test = data["y_test"]

    #Normalization: normalize the whole data the same way, so that dynamic range is fully used 
    #both in and output uses range -1 to +1
    x_complete = np.concatenate((data["x_train"],data["x_valid"],data["x_test"]))
    y_complete= np.concatenate((data["y_train"],data["y_valid"],data["y_test"]))

    #transform data between -1 and 1 or -128 and 127
    x_train = x_train/x_complete.max()
    y_train = y_train/y_complete.max()

    x_test = x_test/x_complete.max()
    y_test = y_test/y_complete.max()

    #DUPLICATE x_train and x_test to n_input_channels channels (then add dithering indipendently)
    x_train = np.repeat(x_train,n_input_channels,axis=1)
    x_test = np.repeat(x_test,n_input_channels,axis=1)

    # ADD dithering (indipendently to each input channel)
    if args.dither_std > 0:
        print("Adding dither std",args.dither_std ,"hicutoff",args.dither_hicutoff)
        x_train = x_train + create_dithering_noise(x_train.shape,sampling_rate,args.dither_std,args.dither_hicutoff,args.dither_order,args.dither_pdf)
        x_test = x_test + create_dithering_noise(x_test.shape,sampling_rate,args.dither_std,args.dither_hicutoff,args.dither_order,args.dither_pdf)
    
    out_range = 2**args.output_bitdepth #number of possible output values

    #new change: do NOT quantize reference values if not specified by quantize_target
    if args.act_mode_8bit:
        x_train = (x_train*0.5*256.).round().clip(min=-128, max=127)
        if args.quantize_target:
            y_train = (y_train*0.5*out_range).round().clip(min=-out_range//2, max=out_range//2-1) # uses 24 bit to have some headroom
        else:
            y_train = (y_train*0.5*out_range) #.round().clip(min=-out_range//2, max=out_range//2-1) # uses 24 bit to have some headroom

        x_test = (x_test*0.5*256.).round().clip(min=-128, max=127)
        if args.quantize_target:
            y_test = (y_test*0.5*out_range).round().clip(min=-out_range//2, max=out_range//2-1)
        else:
            y_test = (y_test*0.5*out_range)
    else:
        x_train = (x_train*0.5*256.).round().clip(min=-128, max=127)/(128.)
        if args.quantize_target:
            y_train = (y_train*0.5*out_range).round().clip(min=-out_range//2, max=out_range//2-1)/(128.)
        else:
            y_train = (y_train*0.5*out_range)/(128.)

        x_test = (x_test*0.5*256.).round().clip(min=-128, max=127)/(128.)
        if args.quantize_target:
            y_test = (y_test*0.5*out_range).round().clip(min=-out_range//2, max=out_range//2-1)/(128.)
        else:
            y_test = (y_test*0.5*out_range)/(128.)


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


def pedalnet_get_datasets_ich16_och16(data, load_train=True, load_test=True):
    n_input_channels = 16
    n_output_channels = 16
    return pedalnet_get_datasets(n_input_channels,n_output_channels,data, load_train, load_test)

datasets = [
    {
        'name': 'PEDALNET_ICH16_OCH16',
        'input': (16, sample_size), #1 channel and 1D
        #'output': list(map(str, range(10))), labels: only for NOT regression
        'output': [1], #WHY do I need to put this shit here...
        'loader': pedalnet_get_datasets_ich16_och16,
        'regression': True 
    },
]
