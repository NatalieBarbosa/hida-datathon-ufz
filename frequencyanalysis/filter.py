# -*- coding: utf-8 -*-
"""
Some filter for signla processing.
"""

from scipy.signal import butter, lfilter



def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    """
    Filters a specific range of frequencies from the signal.

    Parameters
    ----------

    data : 1D-array
        Dataset
    lowcut : float
        Lower frequency in Hz. Ex.: 2 years = 1 / ( 365 * 86400 * 2) [Hz]
    highcut : float
        Higher frequency in Hz
    fs : float
        Sampling frequency [Hz]
    order : int
        The order of the filter.

    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
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
