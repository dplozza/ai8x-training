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

import numpy as np

import pickle
from matplotlib import pyplot as plt
import json
import argparse

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

#GLOBAL SHIT, done once file is loaded

supported_models = []
supported_sources = []
model_names = []
dataset_names = []

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


def load_args(model_log_path,act_mode_8bit=False):

    #create args
    cmd_line_args = "--model ai85net5 --dataset MNIST"
    args = parsecmd.get_parser(model_names, dataset_names).parse_args(cmd_line_args.split())

    with open(model_log_path+'configs/commandline_args.txt', 'r') as f:
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

def load_model(load_model_path,args):

    qat_policy = args.qat_policy_dict
    
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


def OBSOLETE_load_models_datasets():

    # Load all models and datasets
    supported_models = []
    supported_sources = []
    model_names = []
    dataset_names = []

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

    return supported_models, supported_sources,model_names,dataset_names
 
    
