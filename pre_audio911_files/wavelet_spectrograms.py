# import libraries
import numpy as np
from scipy import signal, ndimage
from scipy.interpolate import interp1d

import pywt

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize, LogNorm, NoNorm
# from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import time

import librosa

# %matplotlib inline
# plt.rc('figure', figsize=(16, 4))

#load audio data
import librosa

sr = 8000
_wav_file_ = "/home/riccardopasini/Documents/Aclai/Julia_additional_files/test.wav"
(wav_data, sampling_frequency) = librosa.load(_wav_file_, sr=sr, mono=True)



n_samples = len(wav_data)
total_duration = n_samples / sampling_frequency
sample_times = np.linspace(0, total_duration, n_samples)
# numpy.linspace(start, stop, num)
# Returns num evenly spaced samples, calculated over the interval [start, stop].
# num: Number of samples to generate.


# A Plotting Function
def spectrogram_plot(z, times, frequencies, coif, cmap=None, norm=Normalize(), ax=None, colorbar=True):
    ###########################################################################
    # plot
    
    # set default colormap, if none specified
    if cmap is None:
        cmap = plt.colormaps('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = plt.colormaps.get_cmap(cmap)

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    xx,yy = np.meshgrid(times,frequencies)
    ZZ = z
    
    im = ax.pcolor(xx,yy,ZZ, norm=norm, cmap=cmap)
    ax.plot(times,coif)
    ax.fill_between(times,coif, step="mid", alpha=0.4)
    
    if colorbar:
        cbaxes = inset_axes(ax, width="2%", height="90%", loc=4) 
        fig.colorbar(im,cax=cbaxes, orientation='vertical')

    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())

    return ax

# For comparison - the Short Time Fourier Transform Spectrogram
def stft_gaussian_spectrogram(x, fs, window_dur=0.005, step_dur=None, detrend=True, normalize=True):
# modified from: https://github.com/drammock/spectrogram-tutorial/blob/master/spectrogram.ipynb
    
    ###########################################################################
    # detrend and normalize
    if detrend:
        x = signal.detrend(x,type='linear')
    if normalize:
        stddev = x.std()
        x = x / stddev

    ###########################################################################
    # set default for step_dur, if unspecified. This value is optimal for Gaussian windows.
    if step_dur is None:
        step_dur = window_dur / np.sqrt(np.pi) / 8.
    
    ###########################################################################
    # convert window & step durations from seconds to numbers of samples (which is what
    # scipy.signal.spectrogram takes as input).
    window_nsamp = int(window_dur * fs * 2)
    step_nsamp = int(step_dur * fs)
    
    ###########################################################################
    # make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
    # backward from window_nsamp we can calculate σ.
    window_sigma = (window_nsamp + 1) / 6
    window = signal.gaussian(window_nsamp, window_sigma)
    
    ###########################################################################
    # convert step size into number of overlapping samples in adjacent analysis frames
    noverlap = window_nsamp - step_nsamp
    
    ###########################################################################
    # compute the power spectral density
    freqs, times, power = signal.spectrogram(x, detrend=False, mode='psd', fs=fs,
                                      scaling='spectrum', noverlap=noverlap,
                                      window=window, nperseg=window_nsamp)

    # smooth a bit
    power = ndimage.gaussian_filter(power, sigma=2)

    return power, times, freqs


# Continuous Wavelet Transform Spectrogram
def cwt_spectrogram(x, fs, nNotes=12, detrend=True, normalize=True):
    
    N = len(x)
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x = signal.detrend(x,type='linear')
    if normalize:
        stddev = x.std()
        x = x / stddev

    ###########################################################################
    # Define some parameters of our wavelet analysis. 

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)
    
#     print (scales)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef, freqs=pywt.cwt(x,scales,'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # power
#     power = np.abs(coef)**2
    power = np.abs(coef * np.conj(coef))
    
    # smooth a bit
    power = ndimage.gaussian_filter(power, sigma=2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return power, times, frequencies, coif


# Cross Wavelet Transform Spectrogram
def xwt_spectrogram(x1, x2, fs, nNotes=12, detrend=True, normalize=True):
    
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2),   "error: arrays not same size"
    
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x1 = signal.detrend(x1,type='linear')
        x2 = signal.detrend(x2,type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2

    ###########################################################################
    # Define some parameters of our wavelet analysis. 

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef1, freqs1=pywt.cwt(x1,scales,'cmor1.5-1.0')
    coef2, freqs2=pywt.cwt(x2,scales,'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # Calculates the cross CWT of xs1 and xs2.
    coef12 = coef1 * np.conj(coef2)

    ###########################################################################
    # power
    power = np.abs(coef12)

    # smooth a bit
    power = ndimage.gaussian_filter(power, sigma=2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return power, times, frequencies, coif


# Cross Wavelet Transform Phase
def xwt_phase(x1, x2, fs, nNotes=12, detrend=True, normalize=True):
    
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2),   "error: arrays not same size"
    
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x1 = signal.detrend(x1,type='linear')
        x2 = signal.detrend(x2,type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2

    ###########################################################################
    # Define some parameters of our wavelet analysis. 

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef1, freqs1=pywt.cwt(x1,scales,'cmor1.5-1.0')
    coef2, freqs2=pywt.cwt(x2,scales,'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # Calculate the cross transform of xs1 and xs2.
    coef12 = coef1 * np.conj(coef2)

    phase = np.angle(coef12)

    # smooth a bit
    phase = ndimage.gaussian_filter(phase, sigma=2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return phase, times, frequencies, coif


# Cross Wavelet Transform Coherence
def xwt_coherence(x1, x2, fs, nNotes=12, detrend=True, normalize=True):
    
    N1 = len(x1)
    N2 = len(x2)
    assert (N1 == N2),   "error: arrays not same size"
    
    N = N1
    dt = 1.0 / fs
    times = np.arange(N) * dt

    ###########################################################################
    # detrend and normalize
    if detrend:
        x1 = signal.detrend(x1,type='linear')
        x2 = signal.detrend(x2,type='linear')
    if normalize:
        stddev1 = x1.std()
        x1 = x1 / stddev1
        stddev2 = x2.std()
        x2 = x2 / stddev2

    ###########################################################################
    # Define some parameters of our wavelet analysis. 

    # maximum range of scales that makes sense
    # min = 2 ... Nyquist frequency
    # max = np.floor(N/2)

    nOctaves = np.int(np.log2(2*np.floor(N/2.0)))
    scales = 2**np.arange(1, nOctaves, 1.0/nNotes)

    ###########################################################################
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=1.5 and center frequency of 1.0
    coef1, freqs1=pywt.cwt(x1,scales,'cmor1.5-1.0')
    coef2, freqs2=pywt.cwt(x2,scales,'cmor1.5-1.0')
    frequencies = pywt.scale2frequency('cmor1.5-1.0', scales) / dt
    
    ###########################################################################
    # Calculates the cross transform of xs1 and xs2.
    coef12 = coef1 * np.conj(coef2)

    ###########################################################################
    # coherence
    scaleMatrix = np.ones([1, N]) * scales[:, None]
    S1 = ndimage.gaussian_filter((np.abs(coef1)**2 / scaleMatrix), sigma=2)
    S2 = ndimage.gaussian_filter((np.abs(coef2)**2 / scaleMatrix), sigma=2)
    S12 = ndimage.gaussian_filter((np.abs(coef12 / scaleMatrix)), sigma=2)
    WCT = S12**2 / (S1 * S2)

    ###########################################################################
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2*np.pi
    cmor_coi = 1.0 / np.sqrt(2)
    cmor_flambda = 4*np.pi / (f0 + np.sqrt(2 + f0**2))
    # cone of influence in terms of wavelength
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = cmor_flambda * cmor_coi * dt * coi
    # cone of influence in terms of frequency
    coif = 1.0/coi


    return WCT, times, frequencies, coif

##################################################################################################àà


# Test on some data
plt.rcParams['figure.figsize'] = (16, 6)

# STFT Spectrogram
###########################################################################
# calculate spectrogram

wLen = 50/sampling_frequency
print(wLen)

t0 = time.time()
power, times, frequencies = stft_gaussian_spectrogram(wav_data, sampling_frequency, window_dur=wLen)
print (time.time()-t0)
coif = np.zeros(times.shape)

###########################################################################
# plot

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(sample_times, wav_data, color='b');

ax1.set_xlim(0, total_duration)
ax1.set_xlabel('time (s)')
ax1.grid(True)

spectrogram_plot(power, times, frequencies, coif, cmap='jet', norm=LogNorm(), ax=ax2)

ax2.set_xlim(0, total_duration)
# ax2.set_ylim(0, 0.5*sampling_frequency)
ax2.set_ylim(2.0/total_duration, 0.5*sampling_frequency)
ax2.set_xlabel('time (s)')
ax2.set_ylabel('frequency (Hz)');

ax2.grid(True)

# ax2.set_yscale('log')





