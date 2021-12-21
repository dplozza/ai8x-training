import fnmatch
import os
import sys
import time

import numpy as np

import torch

import json
import pickle

from numpy import pi, convolve
from scipy.signal.filter_design import bilinear
from scipy.signal import lfilter
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import scipy as sp

from matplotlib import pyplot as plt

#----------
#FILTERING
#----------

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype='highpass')
    return b, a
def butter_highpass_filter(data, lowcut, fs, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, [high], btype='lowpass')
    return b, a
def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def A_weighting(Fs):
    """Design of an A-weighting filter.
    
    Source: https://gist.github.com/endolith/148112/5ac0416df2a709489b4bd84ea1393893d2449139
    
    B, A = A_weighting(Fs) designs a digital A-weighting filter for 
    sampling frequency Fs. Usage: y = lfilter(B, A, x).
    Warning: Fs should normally be higher than 20 kHz. For example, 
    Fs = 48000 yields a class 1-compliant filter.
    
    Originally a MATLAB script. Also included ASPEC, CDSGN, CSPEC.
    
    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
            couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.
    
    http://www.mathworks.com/matlabcentral/fileexchange/69
    http://replaygain.hydrogenaudio.org/mfiles/adsgn.m
    Translated from adsgn.m to PyLab 2009-07-14 endolith@gmail.com
    
    References: 
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.
    
    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997
    
    NUMs = [(2 * pi * f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
    DENs = convolve([1, +4 * pi * f4, (2 * pi * f4)**2], 
                    [1, +4 * pi * f1, (2 * pi * f1)**2], mode='full')
    DENs = convolve(convolve(DENs, [1, 2 * pi * f3], mode='full'), 
                                   [1, 2 * pi * f2], mode='full')
    
    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, Fs)

def de_emphasis_filter(x,coeff=0.95):
    a = [1, -coeff]
    b = [1, 0]
    return lfilter(b, a,x)

def pre_emphasis_aweight_filter(x,sr):
    #A-weighting filter
    sr = 44100
    b,a = A_weighting(sr)
    return lfilter(b, a,x)

def pre_emphasis_lowpass_aweight_filter(x,sr):
    #A-weighting filter low pass filtered https://ieeexplore.ieee.org/abstract/document/9052944
    x = pre_emphasis_aweight_filter(x,sr)
    filt_coeff = 0.85
    a = [1,  0]
    b = [1, filt_coeff]
    return lfilter(b, a,x)


#------------
#ESR LOSSES
#------------

def error_to_signal(y, y_pred,pre_filter_coeff,regularizer=1e-10,ignore_zeros=False,zero_treshold=1e-6):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    if pre_filter_coeff!=0:
        y, y_pred = pre_emphasis_filter(y,pre_filter_coeff), pre_emphasis_filter(y_pred,pre_filter_coeff)
    if ignore_zeros: #set y_pred value to y when y is "close" to zero, so that that terms don't count for loss
        y_nonzero = np.abs(y) < zero_treshold
        #y = y[y_nonzero].reshape(y.shape)
        #y_pred_filt = y_pred[y_nonzero]
        y_pred_filt = y_pred.clone()
        y_pred_filt[y_nonzero] = y[y_nonzero]
        
        return (y - y_pred_filt).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + regularizer)
        
    return (y - y_pred).pow(2).sum(dim=2) / (y.pow(2).sum(dim=2) + regularizer)

def error_to_signal_aweight(y, y_pred,sr,regularizer=1e-10):
    y, y_pred = pre_emphasis_aweight_filter(y,sr), pre_emphasis_aweight_filter(y_pred,sr)
    return error_to_signal(torch.tensor(y),torch.tensor(y_pred),pre_filter_coeff=0,regularizer=regularizer)

def error_to_signal_lowpass_aweight(y, y_pred,sr,regularizer=1e-10):
    y, y_pred = pre_emphasis_lowpass_aweight_filter(y,sr), pre_emphasis_lowpass_aweight_filter(y_pred,sr)
    return error_to_signal(torch.tensor(y),torch.tensor(y_pred),pre_filter_coeff=0,regularizer=regularizer)

def pre_emphasis_filter(x, coeff=0.95):
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)

#----------
#DITHERING
#----------

def create_dithering_noise(shape,sr,std,lowcut,order=1,pdf="normal"):
    """Creates dithering noises of the given shape, to be added to  signal
    Args:
        shape: shape of the array
        sr: sampling rate
        std: in normal case, std deviation with respect to quant level, in triangualr case is max noise value
        lowcut: cutoff freq of low pass filter on dithering noise
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