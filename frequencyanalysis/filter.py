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
        Lower frequency in Hz
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
