#!/usr/bin/env python
# coding: utf-8

# In[41]:


"""
Provide functions used in features implementation
Provides: frame auduio

Author: Ren Yuan (Peter) Xue
"""

import numpy as np

def get_num_fft(sample_rate, frame_size):
    """
    Function get_num_fft calculates optimal number of FFT points based on frame length.
    Less number of FFT points than length of frame
    will lose precision by droppping many of the samples.
    Therefore, we want num_fft as a power of 2, greater than frame length.
    
    @param sample_rate: The sample rate of audio signal we working with.
    @param frame_size: Time interval we are taking within frames.
    @returns: Optimal number of FFT points.
    """
    frame_length = sample_rate * frame_size
    num_fft = 1
    while num_fft < frame_length:
        num_fft *= 2
    return num_fft

def frame_audio(signal, sample_rate, frame_size, frame_step, num_fft):
    """
    Function framing devides signal into small intervals of size fsize.
    
    @param signal: Audio signal we are working with.
    @param sample_rate: The sample rate of our audio signal.
    @param frame_size: Time interval we are taking within frames.
    @param frame_step: Time step we are taking between frames.
    @param num_fft: Number of FFT points.
    @returns: A (frames * length of frame) matrix.
    """
    # Convert from seconds to samples.
    frame_length, frame_stride = frame_size * sample_rate, frame_step * sample_rate
    frame_length, frame_stride = int(round(frame_length)), int(round(frame_stride))
    signal_length = len(signal)
    # Make sure that we have at least 1 frame.
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_stride))
    pad_signal_length = num_frames * frame_stride + frame_length
    diff = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, diff)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

def get_filter(freq_min, freq_max, num_mel_filter, num_fft, sample_rate):
    """
    Function get_filter_points calculates where the filters in filter bank locate
    
    @param freq_min: Lowest frequency band edge of Mel filters in Hz.
    @param freq_max: Highest frequency band edge of Mel filters in Hz.
    @param num_mel_filter: Number of filter points in filter banks on Mel scale.
    @param num_fft: Number of FFT points.
    @param sample_rate: The sample rate of audio signal we working with.
    @returns: Filters used for computing filter bank feature.
    """
    # Get filter points.
    freq_min_mel = hz2mel(freq_min)
    freq_max_mel = hz2mel(freq_max)
    mels = np.linspace(freq_min_mel, freq_max_mel, num=num_mel_filter+2)
    freqs = mel2hz(mels)
    filter_points = np.floor((num_fft + 1) / sample_rate * freqs).astype(int)
    # Get filter bank filters.
    filters = np.zeros((len(filter_points)-2, int(num_fft/2+1)))
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n+1]] = np.linspace(0, 1, filter_points[n+1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n+2] - filter_points[n+1])
    return filters

def dct_3(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num, filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)   
    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)      
    return basis

def pre_emphasis(signal, coef):
    """
    Function pre-emphasis applies pre-emphasis filter 
    on the signal to amplify the high frequencies.
    
    @param signal: Audio signal.
    @param coef: Coefficient used in pre-empahsis filter.
    @returns: Pre-emphasized signal after applying the filter.
    """
    return np.append(signal[0], signal[1:] - coef * signal[:-1])

def hz2mel(freq):
    """
    Function hz2mel calculates Mel values.
    
    @param freq: Frequency.
    @returns: Corresponding Mel value for given frequency.
    """
    return 2595.0 * np.log10(1.0 + freq / 700.0)
    
def mel2hz(mels):
    """
    Function mel2hz calculates Hertz values.
    
    @param mel: Mel value.
    @returns: Corresponding Hertz value for given Mel value.
    """
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

