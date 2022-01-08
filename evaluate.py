# Adapted from PedalNetRT

"""
Evaluation script for wavenet model

USAGE:
    First call args = evaluate.load_args(model_log_path),
    Then load separate model files individually with 
        evaluate.load_model(load_model_path_1,args)
        evaluate.load_model(load_model_path_2,args) etc...


    model_log_path = path of log folder (ex. "./logs/model_log/")
    load_model_path_1 = path of specific model checkpoint file (ex. "./logs/model_log/model_qat_best.pth.tar")
        (Model usually should be iniside log folder)
"""

import fnmatch
import os
import sys
import time
import traceback
from pydoc import locate
import re

import numpy as np

import pickle
from matplotlib import pyplot as plt
import json
import argparse
from scipy.io import wavfile

# TensorFlow 2.x compatibility
try:
    import tensorboard  # pylint: disable=import-error
    import tensorflow  # pylint: disable=import-error
    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
except (ModuleNotFoundError, AttributeError):
    pass

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data

import distiller.apputils as apputils

# pylint: enable=no-name-in-module
import ai8x
import datasets
import parse_qat_yaml
import parsecmd

from train_test import create_model, update_old_model_params

from evaluate_utils import *

#--------------
# GLOBALS
#--------------

# GLOBAL OPERATIONS, done once file is loaded:
# dynamically get list of models and datasets
supported_models = []
supported_sources = []
model_names = []
dataset_names = []



#---------------
#MODEL LOADING
#---------------

def load_args(model_log_path,act_mode_8bit=False):

    #load models and datasets globally
    _load_models_datasets_globally()

    #create args
    cmd_line_args = "--model ai85net5 --dataset MNIST"
    args = parsecmd.get_parser(model_names, dataset_names).parse_args(cmd_line_args.split())

    with open(model_log_path+'\\configs\\commandline_args.txt', 'r') as f:
        loaded_args = json.load(f)
    args.num_hidden_channels = loaded_args["num_hidden_channels"]
    print("num hidden channels",args.num_hidden_channels)

    for key in loaded_args.keys():
        if key == "load_model_path":
            continue
        args.__dict__[key] = loaded_args[key]
    args.ai8x_device = args.device

    args.cnn = loaded_args["cnn"]
    print("model",args.cnn)
    print("dataset",args.dataset)
    print("dl",args.dilation_depth,"  p",args.dilation_power,"  r",args.num_repeat,"  k",args.kernel_size)

    #dithering
    if "dither_std" not in loaded_args.keys():
        args.dither_std = 0
    else:
        args.dither_std = loaded_args["dither_std"]
        args.dither_hicutoff = loaded_args["dither_hicutoff"]
    print("Dither std:",args.dither_std,"hicutoff:",args.dither_hicutoff)

    args.act_mode_8bit = act_mode_8bit

    #INIT stuff
    _init_stuff(args)

    return args

def load_model(load_model_path,args,act_mode_8bit=False):

    qat_policy = args.qat_policy_dict
    
    args.act_mode_8bit=act_mode_8bit
    _init_stuff(args)
    
    model = create_model(supported_models, args.dimensions, args)
    
    update_old_model_params(load_model_path, model)
    if qat_policy is not None:
        checkpoint = torch.load(load_model_path,
                                map_location=lambda storage, loc: storage)
        if checkpoint.get('epoch', None) >= qat_policy['start_epoch']:
            ai8x.fuse_bn_layers(model)
    model = apputils.load_lean_checkpoint(model, load_model_path,
                                          model_device=args.device)
    ai8x.update_model(model)

    model.eval()
    print("Model loaded correctly")
    return model

def set_8bit_mode(act_mode_8bit,args):
    args.act_mode_8bit = act_mode_8bit
    ai8x.set_device(args.ai8x_device, args.act_mode_8bit, args.avg_pool_rounding)

def _init_stuff(args):
    
    ai8x.set_device(args.ai8x_device, args.act_mode_8bit, args.avg_pool_rounding)


    if args.cpu or not torch.cuda.is_available():
        if not args.cpu:
            print("WARNING: CUDA hardware acceleration is not available, training will be slow") # Print warning if no hardware acceleration
        # Set GPU index to -1 if using CPU
        args.device = 'cpu'
        args.gpus = -1
    else:
        args.device = 'cuda'
        if args.gpus is not None:
            try:
                args.gpus = [int(s) for s in args.gpus.split(',')]
            except ValueError as exc:
                raise ValueError('ERROR: Argument --gpus must be a comma-separated '
                                 'list of integers only') from exc
            available_gpus = torch.cuda.device_count()
            for dev_id in args.gpus:
                if dev_id >= available_gpus:
                    raise ValueError('ERROR: GPU device ID {0} requested, but only {1} '
                                     'devices available'
                                     .format(dev_id, available_gpus))
            # Set default device in case the first one on the list != 0
            torch.cuda.set_device(args.gpus[0])


    # DCOM dataset selection (regression vs classification)
    selected_source = next((item for item in supported_sources if item['name'] == args.dataset))
    args.labels = selected_source['output']
    args.num_classes = len(args.labels)
    if args.num_classes == 1 \
       or ('regression' in selected_source and selected_source['regression']):
        args.regression = True
    dimensions = selected_source['input']
    if len(dimensions) == 2:
        dimensions += (1, )
    args.dimensions = dimensions

    args.datasets_fn = selected_source['loader']


    # Get policy for quantization aware training
    qat_policy = parse_qat_yaml.parse(args.qat_policy) \
        if args.qat_policy.lower() != "none" else None

    # Get policy for once for all training policy
    nas_policy = parse_nas_yaml.parse(args.nas_policy) \
        if args.nas and args.nas_policy.lower() != '' else None

    #if not args.load_serialized and args.gpus != -1 and torch.cuda.device_count() > 1:
    #    model = torch.nn.DataParallel(model, device_ids=args.gpus).to(args.device)
    
    args.qat_policy_dict = qat_policy

def _load_models_datasets_globally():

    # Load all models and datasets and store in module GLOBAL lists!
    global supported_models
    global supported_sources
    global model_names
    global dataset_names

    #lists must exists, and are initialized only once module is loaded one time
    #this is kinda the sigleton programming pattern, and it's BAD, but it works
    if len(supported_models) == 0 and len(supported_sources) == 0: #do only once

        # Dynamically load models
        for _, _, files in sorted(os.walk('models')):
            for name in sorted(files):
                if fnmatch.fnmatch(name, '*.py'):
                    fn = 'models.' + name[:-3]
                    m = locate(fn)
                    try:
                        for i in m.models:
                            i['module'] = fn
                        supported_models += m.models
                        model_names += [item['name'] for item in m.models]
                    except AttributeError:
                        # Skip files that don't have 'models' or 'models.name'
                        pass

        # Dynamically load datasets
        for _, _, files in sorted(os.walk('datasets')):
            for name in sorted(files):
                if fnmatch.fnmatch(name, '*.py'):
                    ds = locate('datasets.' + name[:-3])
                    try:
                        supported_sources += ds.datasets
                        dataset_names += [item['name'] for item in ds.datasets]
                    except AttributeError:
                        # Skip files that don't have 'datasets' or 'datasets.name'
                        pass


#------------------
#DATASET LOADING
#------------------

def prepare_test_set(args,n_samples=None,sr=44100,dither_zeros=True):
    """
    Args:
        n_samples: if None, load all test samples, else load only first n_samples test samples
        dither_zeros: if true add dithering also to zeros. Else only add dithering to nonzero input values
    """
    try:
        dataset_name = args.dataset.split("_CH")[-2]
    except:
        dataset_name =  args.dataset
        
    if dataset_name == "PEDALNET":
        data = pickle.load(open("data/PEDALNET/ts9_test1_FP32.pickle", "rb"))
    elif dataset_name == "PEDALNET2":
        data = pickle.load(open("data/PEDALNET/ts9_test2_sz512.pickle", "rb"))
    elif dataset_name == "PEDALNET_BIGMUFF1":
        data = pickle.load(open("data/PEDALNET/big_muff_1.pickle", "rb"))
    elif dataset_name == "PEDALNET_TUBECLONE_BG": 
        data = pickle.load(open("data/PEDALNET/tube_clone_bassguitar.pickle", "rb"))
    elif dataset_name == "PEDALNET_TUBECLONE_G": 
        data = pickle.load(open("data/PEDALNET/tube_clone_guitar.pickle", "rb"))
    elif dataset_name == "PEDALNET_TUBECLONE_GSK":    
        data = pickle.load(open("data/PEDALNET/tube_clone_guitar_sk.pickle", "rb"))

    elif dataset_name == "PEDALNET_TS808_G": 
        data = pickle.load(open("data/PEDALNET/ts808_guitar.pickle", "rb"))
    elif dataset_name == "PEDALNET_TS808_GSK": 
        data = pickle.load(open("data/PEDALNET/ts808_guitar_sk.pickle", "rb"))

    elif dataset_name == "PEDALNET_GLOVE_G": 
        data = pickle.load(open("data/PEDALNET/glove_guitar.pickle", "rb"))
    elif dataset_name == "PEDALNET_GLOVE_GSK": 
        data = pickle.load(open("data/PEDALNET/glove_guitar_sk.pickle", "rb"))

    elif dataset_name == "PEDALNET_BIGMUFF_G": 
        data = pickle.load(open("data/PEDALNET/big_muff_pi_guitar.pickle", "rb"))
    elif dataset_name == "PEDALNET_BIGMUFF_GSK": 
        data = pickle.load(open("data/PEDALNET/big_muff_pi_guitar_sk.pickle", "rb"))


    elif dataset_name == "PEDALNET_FUZZ":
        data = pickle.load(open("data/PEDALNET/fuzz.pickle", "rb"))
    elif dataset_name == "PEDALNET_DIST":
        data = pickle.load(open("data/PEDALNET/dist.pickle", "rb"))
    elif dataset_name == "PEDALNET_AMPCAB1":
        data = pickle.load(open("data/PEDALNET/ampli_cab_1.pickle", "rb"))
    elif args.dataset in ["PEDALNET_15XOS","PEDALNET_15XOS_FLOAT"]:
        data = pickle.load(open("data/AUDIO/data_1.5xos.pickle", "rb"))
    elif args.dataset in ["PEDALNET_2XOS","PEDALNET_2XOS_FLOAT"]:
        data = pickle.load(open("data/AUDIO/data_2xos.pickle", "rb"))
    else:
        raise Exception("Unknown dataset",args.dataset)


    x_train = np.concatenate((data["x_train"],data["x_valid"]))
    y_train = np.concatenate((data["y_train"],data["y_valid"]))

    x_test = data["x_test"] 
    y_test = data["y_test"] 
    
    x_max_orig = np.concatenate((x_train,x_test)).max()
    y_max_orig = np.concatenate((y_train,y_test)).max()
    
    x_test_orig = x_test/x_max_orig
    y_test_orig = y_test/y_max_orig
    
    #PREPROCESS inptut data with pre-emph filter!
    if args.preprocess_filter > 0:
        filter_coeff = args.preprocess_filter
        x_train = pre_emphasis_filter(torch.tensor(x_train.reshape(1,1,-1)),filter_coeff).numpy().reshape(x_train.shape)
        y_train = pre_emphasis_filter(torch.tensor(y_train.reshape(1,1,-1)),filter_coeff).numpy().reshape(y_train.shape)
        #assume sample batches are NOT consecutive: first sample of fir output is INVALID -> replace with second sample, for each sample!
        #necessary to avoid high freq spikes (which interfere with the normalization)
        x_train[:,:,0] = x_train[:,:,1]
        y_train[:,:,0] = y_train[:,:,1]

        x_test = pre_emphasis_filter(torch.tensor(x_test.reshape(1,1,-1)),filter_coeff).numpy().reshape(x_test.shape)
        y_test = pre_emphasis_filter(torch.tensor(y_test.reshape(1,1,-1)),filter_coeff).numpy().reshape(y_test.shape)
        #test audio is assumed to be consecutive samples, we only have to correct fisrt sample of first batch
        x_test[0,:,0] = x_test[0,:,1]
        y_test[0,:,0] = y_test[0,:,1]

    #Normalization: normalize the whole data the same way, so that dynamic range is fully used  both in and output uses range -1 to +1
    x_complete = np.concatenate((x_train,x_test))
    y_complete= np.concatenate((y_train,y_test))
    
    x_max = x_complete.max()
    y_max = y_complete.max()

    #transform data between -1 and 1
    x_train = x_train/x_max
    y_train = y_train/y_max

    x_test = x_test/x_max
    y_test = y_test/y_max   
    
    #MULTICHANNEL
    n_input_channels = 1
    
    if len(args.dataset.split("CH"))>1:
        n_input_channels =  int(args.dataset.split("CH")[-1])
    # if args.dataset in ["PEDALNET_CH12","PEDALNET2_CH12"]:
    #     n_input_channels = 12
    # elif args.dataset in ["PEDALNET_CH16","PEDALNET2_CH16"]:
    #     n_input_channels = 16
    # elif args.dataset in ["PEDALNET_CH24","PEDALNET2_CH24"]:
    #     n_input_channels = 24
    # elif args.dataset in ["PEDALNET_CH32","PEDALNET2_CH32"]:
    #     n_input_channels = 32
    # elif args.dataset in ["PEDALNET_CH64","PEDALNET2_CH64"]:
    #     n_input_channels = 64
    x_test = np.repeat(x_test,n_input_channels,axis=1)


    #quantization
    quantize = True
    if not args.dataset=="PEDALNET_FLOAT" and quantize:
        # DITHERING
        print("dither pdf:",args.dither_pdf,"dither std:",args.dither_std,"dither order:",args.dither_order,"dither hicutoff:",args.dither_hicutoff)
        if args.dither_std > 0:
            x_test = x_test + create_dithering_noise(x_test.shape,sr,args.dither_std,args.dither_hicutoff,args.dither_order,args.dither_pdf)

        x_test = (x_test*0.5*256.).round().clip(min=-128, max=127)/(128.)
        #y_test = (y_test*0.5*256.).round().clip(min=-128, max=127)/(128.)

    
    #take only first n_samples
    if n_samples is not None:
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
        x_test_orig = x_test_orig[:n_samples]
        y_test_orig = y_test_orig[:n_samples]

    prev_sample = np.concatenate((np.zeros_like(x_test[0:1]), x_test[:-1]), axis=0)
    pad_x_test = np.concatenate((prev_sample, x_test), axis=2)

    #print("x_test shape ",x_test.shape)
    #print("pad_x_test shape",pad_x_test.shape)
    #plt.plot(x_test.flatten()[100:200])
    
    dataset_inf = {}
    dataset_inf['sampling_rate'] = sr
    dataset_inf['x_max'] = x_max
    dataset_inf['y_max'] = y_max
    dataset_inf['x_max_orig'] = x_max_orig
    dataset_inf['y_max_orig'] = y_max_orig
    dataset_inf['x_test'] = x_test
    dataset_inf['y_test'] = y_test
    dataset_inf['x_test_orig'] = x_test_orig
    dataset_inf['y_test_orig'] = y_test_orig

    dataset_inf['pad_x_test'] = pad_x_test


    return dataset_inf


#---------------------------
#PREDICTION AND EVALUATION
#---------------------------

def evaluate_model(model_path,sampling_rate=44100,save_seconds=10):
    """
    Evaluate quantized (_qat_best_q.pth.tar) model in log folder "model_path", and store results in that folder
    Args:   
        model_path: path of the log folder with model inside (_qat_best_q.pth.tar) (ending with slash)
        sampling_rate: the sampling rate of the training audio
        save_samples: number of predicted audio seconds to save, if none save all
    """

    n_samples = None
    act_mode_8bit = True
    y_pred_qat,dataset_inf,ckpt_inf,args = predict_test_set(model_path,act_mode_8bit,n_samples,sampling_rate)
    y_test = dataset_inf["y_test"]

    (y_test_filt,y_pred_qat_filt) = de_emphasize(y_pred_qat,dataset_inf,args)
    scores_dict = get_scores(y_test,y_test_filt,y_pred_qat,y_pred_qat_filt,ckpt_inf,args,sampling_rate)

    #save data in human readable txt
    decimals = 5
    with open(model_path+'\\score.txt', 'w') as f:
            f.write('Best epoch:   ' + str(scores_dict['epoch_qat']) + '\n')
            f.write('ESR_train:    ' + str(round(scores_dict['ESR_train'], decimals)) + '\n')
            f.write('ESR_prefilt:  ' + str(round(scores_dict['ESR_prefilt'], decimals)) + '\n')
            f.write('ESR_nofilt:   ' + str(round(scores_dict['ESR_nofilt'], decimals)) + '\n')
            f.write('ESR_aweight:  ' + str(round(scores_dict['ESR_aweight'], decimals)) + '\n')
            f.write('ESR_laweight: ' + str(round(scores_dict['ESR_laweight'], decimals)) + '\n')

    #save dict as jason
    json.dump( scores_dict, open(model_path+"\\score.json", 'w' ) )
    #json.load(open(model_path+"\\score.json", 'r' ))

    #create and save plots
    plot_signals(y_test_filt,y_pred_qat_filt,dataset_inf,scores_dict,model_path)

    #save predicted samples
    save_audio(y_pred_qat_filt,save_seconds,sampling_rate,model_path)

    #print('\n')
    print('Best epoch:   ' + str(scores_dict['epoch_qat']))
    print('ESR_train:    ' + str(round(scores_dict['ESR_train'], decimals)))
    print('ESR_prefilt:  ' + str(round(scores_dict['ESR_prefilt'], decimals)))
    print('ESR_nofilt:   ' + str(round(scores_dict['ESR_nofilt'], decimals)))
    print('ESR_aweight:  ' + str(round(scores_dict['ESR_aweight'], decimals)))
    print('ESR_laweight: ' + str(round(scores_dict['ESR_laweight'], decimals)))
    #print('\n')


def predict_test_set(model_path,act_mode_8bit,n_samples=None,sampling_rate=44100,model_name=None,last_chkpt=False,dither_info=None):
    """
    Load and predict test set for quantized model (qat_best_q) or nonquantized (qat_best) in the given folder
    Args:
        model_path: path of the log folder with model inside (_qat_best_q.pth.tar or _qat_best.pth.tar) (ending with NO slash)
        act_mode_8bit: if True simulate 8 bits (automatically load _q model) else load nonquanzied model
        model_name: if none use default (_qat_best_q.pth.tar or _qat_best.pth.tar), else use specific name
        last_chkpt: if true and model_name=None, use (_qat_checkpoint_q.pth.tar or _qat_checkpoint.pth.tar)
        dither_info: can be used to force dithering if not None. Has to be a dict with fields [dither_std,dither_pdf,dither_oder,dither_hicutoff]

    Returns:
        (y_pred_qat,dataset_inf,ckpt_inf,args)
            y_pred_qat: prediction on test set (numpy array 3 dim)
            dataset_inf: dict containing dataset informations, no time to write doc..
            ckpt_inf: dict containing checkpoint information, no time to write doc...
            args: loaded args used to train
    """
    if model_name is None:
        if last_chkpt:
            load_model_path_qat = model_path+"\\"+(re.split('(?:[0-9]+).(?:[0-9]+).(?:[0-9]+)-(?:[0-9]+)__', model_path))[-1]+"_qat_checkpoint.pth.tar"
            if act_mode_8bit:
                load_model_path_qat = model_path+"\\"+(re.split('(?:[0-9]+).(?:[0-9]+).(?:[0-9]+)-(?:[0-9]+)__', model_path))[-1]+"_qat_checkpoint_q.pth.tar"
        else:
            load_model_path_qat = model_path+"\\"+(re.split('(?:[0-9]+).(?:[0-9]+).(?:[0-9]+)-(?:[0-9]+)__', model_path))[-1]+"_qat_best.pth.tar"
            if act_mode_8bit:
                load_model_path_qat = model_path+"\\"+(re.split('(?:[0-9]+).(?:[0-9]+).(?:[0-9]+)-(?:[0-9]+)__', model_path))[-1]+"_qat_best_q.pth.tar"
    else:
        load_model_path_qat = model_path +"\\"+model_name
        
    
    args = load_args(model_path,act_mode_8bit)
    if dither_info is not None:
        args.dither_std  = dither_info["dither_std"]
        args.dither_pdf = dither_info["dither_pdf"]
        args.dither_order = dither_info["dither_order"]
        args.dither_hicutoff = dither_info["dither_hicutoff"]

    model_qat = load_model(load_model_path_qat,args,act_mode_8bit)
    
    #checkpont inf
    ckpt_inf = {}
    checkpoint_qat = torch.load(load_model_path_qat,map_location=torch.device('cpu'))
    ckpt_inf["epoch_qat"] = checkpoint_qat["epoch"]
    ckpt_inf["folder_path"] = model_path
    
    #prepare dataset
    dataset_inf = prepare_test_set(args,n_samples,sampling_rate)
    
    #predict QAT (either 8 bits quantized model, or normal float)
    set_8bit_mode(act_mode_8bit,args)
    input_scaling = 1.
    output_scaling = 1.
    if act_mode_8bit:
        input_scaling = 128.
        output_scaling = 2**15
        
    pad_x_test_qat = dataset_inf['pad_x_test']*input_scaling
    y_pred_qat = []
    for x in np.array_split(pad_x_test_qat, 10):
        #y_pred_qat.append(model_qat(torch.from_numpy(x)).detach().numpy()*0.5)
        y_pred_qat.append(model_qat(torch.from_numpy(x)).detach().numpy()/output_scaling)
        
    y_pred_qat = np.concatenate(y_pred_qat)
    y_pred_qat = y_pred_qat[:,  :, -dataset_inf['y_test'].shape[2] :]
    
    return y_pred_qat,dataset_inf,ckpt_inf,args

def de_emphasize(y_pred_qat,dataset_inf,args):
    """
    Get target and prediction in de emphasized form
    Args:
        y_pred_qat_filt: NOT de-filtered (qat) prediction (numpy array 3 dim)   
        args: cmdline training script args
    Return: (y_test_filt,y_pred_qat_filt)
    """
    y_test = dataset_inf["y_test"]
    x_test = dataset_inf["x_test"]
    x_max_orig = dataset_inf["x_max_orig"]
    y_max_orig = dataset_inf["y_max_orig"]
    x_max = dataset_inf["x_max"]
    y_max = dataset_inf["y_max"]
    
    if args.preprocess_filter>0:
        filter_coeff = args.preprocess_filter    
        y_test_filt = dataset_inf['y_test_orig'] #already normalized
        y_pred_qat_filt = de_emphasis_filter(y_pred_qat.flatten(),filter_coeff).reshape(y_pred_qat.shape)
        
        #renormalize (no effect on score, since ESR is normalized by y_test)
        y_pred_qat_filt = y_pred_qat_filt*(y_max/y_max_orig)
    else:
        y_test_filt = y_test
        y_pred_qat_filt = y_pred_qat
    
    return (y_test_filt,y_pred_qat_filt)

def get_scores(y_test,y_test_filt,y_pred_qat,y_pred_qat_filt,ckpt_info,args,sr=44100,regularizer=0):
    """
    Compute scores, return dict with all infos
    Args:   
        y_test: pre-filtered targed (numpy array 3 dim)
        y_test_filt: de-filtered (original!) targed (numpy array 3 dim)
        y_pred_qat: pre-filtered (qat) prediction (numpy array 3 dim)
        y_pred_qat_filt: de-filtered (qat) prediction (numpy array 3 dim)
        ckpt_info: dict with additional ceckpoint info. Keys:
            epoch_qat: checkpoint epoch   
        args: cmdline training script args
    """
    #hyperparameters
    #regularizer=0 #1e-10
    
    scores_dict = {}
    
    #checkpoint info
    scores_dict["epoch_qat"] = ckpt_info['epoch_qat']
    scores_dict["sampling_rate"] = sr
    scores_dict["predicted_samples"] = y_test.flatten().size
    
    #Error to signal, with same conditions as during training: pre-filtered signals and  pre-filter as defined in args
    pre_filter_coeff=args.pre_filter_coeff
    ESR_train = error_to_signal(torch.tensor(y_test).reshape(1,1,-1),torch.tensor(y_pred_qat).reshape(1,1,-1),pre_filter_coeff,regularizer).mean().numpy()
    scores_dict["ESR_train"] = float(ESR_train)
    
    #Error to signal, with pre-filter 0.95
    pre_filter_coeff=0.95
    ESR_prefilt = error_to_signal(torch.tensor(y_test_filt).reshape(1,1,-1),torch.tensor(y_pred_qat_filt).reshape(1,1,-1),pre_filter_coeff,regularizer).mean().numpy()
    scores_dict["ESR_prefilt"] = float(ESR_prefilt)
    
    #Error to signal, no pre-filter
    ESR_nofilt = error_to_signal(torch.tensor(y_test_filt).reshape(1,1,-1),torch.tensor(y_pred_qat_filt).reshape(1,1,-1),0,regularizer,).mean().numpy()
    scores_dict["ESR_nofilt"] = float(ESR_nofilt)
    
    #Error to signal, with A-Weighting pre-filter
    ESR_aweight = error_to_signal_aweight(y_test_filt.reshape(1,1,-1),torch.tensor(y_pred_qat_filt).reshape(1,1,-1),sr).mean().numpy()
    scores_dict["ESR_aweight"] = float(ESR_aweight)
    
    #Error to signal, with lowpassed A-Weighting pre-filter
    ESR_laweight = error_to_signal_lowpass_aweight(y_test_filt.reshape(1,1,-1),torch.tensor(y_pred_qat_filt).reshape(1,1,-1),sr).mean().numpy()
    scores_dict["ESR_laweight"] = float(ESR_laweight)
    
    return scores_dict


def save_audio(y_pred_qat_filt,save_seconds,sr,model_path):
    """Stores first save_samples of y_pred_qat_filt numpy array as audio
    Args:
        save_samples: number of predicted audio seconds to save, if none save all
    """
    if save_seconds is None:
        save_samples = y_pred_qat_filt.flatten().size
    else:
        save_samples = int(save_seconds*sr)
    
    wavfile.write(model_path+"\\y_pred.wav", sr, y_pred_qat_filt.flatten()[:save_samples].astype(np.float32))

    

def plot_signals(y_test_filt,y_pred_qat_filt,dataset_inf,scores_dict,model_path):

    y_test_filt_flat = y_test_filt.flatten()
    y_pred_qat_filt_flat = y_pred_qat_filt.flatten()
    x_test_orig = dataset_inf['x_test_orig'].flatten()
    sr = dataset_inf['sampling_rate']


    Time = np.linspace(0, len(y_test_filt_flat) / sr, num=len(y_test_filt_flat))
    fig, (ax3, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(14, 8))
    fig.suptitle("Predicted vs Actual Signal")
    ax1.plot(Time, y_test_filt_flat, label="reference", color="red")
    fig.tight_layout(pad=3.0)

    Time2 = np.linspace(0, len(y_pred_qat_filt_flat) / sr, num=len(y_pred_qat_filt_flat))
    ax1.plot(Time2, y_pred_qat_filt_flat, label="prediction", color="green")
    ax1.legend(loc="upper right")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Wav File Comparison")
    ax1.grid("on")


    fig.suptitle("Predicted vs Actual Signal (ESR: " + str(round(scores_dict['ESR_nofilt'], 4)) + ")")
    # Plot signal difference
    signal_diff = y_test_filt_flat - y_pred_qat_filt_flat
    ax2.plot(Time2, np.abs(signal_diff), label="signal diff", color="blue")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("abs(pred_signal-actual_signal)")
    ax2.grid("on")


    # Plot the original signal
    Time3 = np.linspace(0, len(x_test_orig) / sr, num=len(x_test_orig))
    ax3.plot(Time3, x_test_orig, label="input", color="purple")
    ax3.legend(loc="upper right")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Original Input")
    ax3.grid("on")

    # Save the plot
    plt.savefig(model_path + "\\signal_comparison_esr_" + str(round(scores_dict['ESR_nofilt'], 4)) + ".png", facecolor="white", edgecolor='none',bbox_inches="tight")

    # Create a zoomed in plot of 0.01 seconds centered at the max input signal value
    sig_temp = x_test_orig.tolist()
    plt.axis(
        [
            Time3[sig_temp.index((max(sig_temp)))] - 0.005,
            Time3[sig_temp.index((max(sig_temp)))] + 0.005,
            min(y_pred_qat_filt_flat),
            max(y_pred_qat_filt_flat),
        ]
    )

    # Save the plot
    plt.savefig(model_path + "\\detail_signal_comparison_esr_" + str(round(scores_dict['ESR_nofilt'], 4)) + ".png",facecolor="white", edgecolor='none', bbox_inches="tight")
    
