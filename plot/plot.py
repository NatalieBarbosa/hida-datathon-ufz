# -*- coding: utf-8 -*-
# from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np


def plot_time_series_for_locations(time_series_list, time, names, **kwargs):
    """
    Parameters
    ----------
    time_series_list : list of array/list
        List containing 1D arrays or lists to plot.
    time : 1D array
        Times.
    names : list
        A list with names for each column.

    Yields
    ------
    A plot with given name.
    """

    for time_series, name in zip(time_series_list, names):
        plt.plot(time, time_series, label=name, **kwargs)
    plt.legend()
    plt.show()
