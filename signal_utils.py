import numpy as np
from scipy.signal import welch
from scipy.signal.windows import hann
from scipy import integrate
from scipy import signal

def digital_filter(x,y,f=1000,order=10,kind="lowpass"):
    dt = x[1]-x[0]
    fs = 1/dt
    sos = signal.butter(order, f, kind, fs=fs, output='sos')
    return signal.sosfiltfilt(sos,y)

def lowpass(x,y,f=1000,order=10):
    return digital_filter(x,y,f=f,order=order,kind="lowpass")

def highpass(x,y,f=1000,order=10,add_average=True):
    temp = digital_filter(x,y,f=f,order=order,kind="highpass")
    const = y.mean() if add_average else 0
    return temp+const

def bandpass(x,y,freq=[99,101],order=10,add_average=True):
    """ freq can be a tuple (fmin,fmax) or a list of tuples if multiple bands """
    freqs = np.atleast_2d(freq)
    temp = np.zeros_like(y)
    for band in freqs:
        temp += digital_filter(x,y,f=band,order=order,kind="bandpass")
    const = y.mean() if add_average else 0
    return temp+const

def bandstop(x,y,freq=[99,101],order=10):
    """ freq can be a tuple (fmin,fmax) or a list of tuples if multiple bands """
    freqs = np.atleast_2d(freq)
    temp = np.zeros_like(y)
    for band in freqs:
        y = digital_filter(x,y,f=band,order=order,kind="bandstop")
    return y

def integrated_rms_noise(x,y,normalize=False,freqs=np.logspace(0,4,100),average_2d=False):
    """ if average_2d: returns only average if y is 2d """
    if normalize: y = y/y.mean()
    noise = [lowpass(x,y,f).std(axis=-1) for f in freqs]
    noise = np.asarray(noise)
    if average_2d and noise.ndim == 2: noise = noise.mean(axis=-1)
    return freqs,noise

def integrated_rms_noise_psd(x,y,normalize=False,average_2d=False,velocity_to_displacement=False,segment_length=None,integration_range=None):
    """ if average_2d: returns only average if y is 2d """
    f,psd = power_spectral_density(x,y,normalize=normalize,average_2d=average_2d,segment_length=segment_length)
    if integration_range is not None:
        idx = (f >= integration_range[0]) & (f <= integration_range[1])
        f = f[idx]
        psd = psd[idx]
    if velocity_to_displacement:
        omega = 2*np.pi*f
        psd = psd/omega/omega
    variance_inegral = integrate.cumtrapz(psd,x=f,initial=0,axis=-1)
    rms_integral = np.sqrt(variance_inegral)
    return f,rms_integral


def power_spectral_density(x,y,normalize=True,average_2d=False,segment_length=None):
    """ if average_2d: returns only average if y is 2d """
    if normalize: y = y/y.mean()
    dt = x[1]-x[0]
    fs = 1/dt
    nperseg = len(x) if segment_length is None else int(segment_length/dt)

    f,psd=welch(y,fs=fs,nperseg=nperseg)
    if average_2d and psd.ndim == 2: psd = asd.mean(axis=0)
    return f,psd
