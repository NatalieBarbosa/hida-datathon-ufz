# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------

def plot_spectrum(
    data,
    frequency,
    name=None,
    labels=None,
    path=None,
    lims=None,
    linestyle="-",
    marker="",
    #markersize=None,
    grid="both",
    unit="[Hz]",
    heading="None",
    figtxt=None,
    comment="",
):
    """
    Function to plot one or multiple power spectra.

    Parameters
    ----------
    data : 2-D array
        Each row represents a seperate power spectrum.
    frequency : 1-D array
        Corresponding frequencies of data.
    name : string
        Name of file. If None, time is used.
    labels : X item list
        Labels for different power spectra as list in same order as data.
    path : string
        Path to store the image.
    lims : list with 2 tuples
        lims[0] = x limit as tuple (xmin,xmax)
        lims[1] = y limit as tuple (ymin,ymax)
        e.g. lims = [(1e-8,1e-4),(1e0,1e5)]
    linestyle : X item list
        List with linestyles for differenct spectra.
    marker : X item list
        List with marker for differenct spectra.
    grid : string
        "major", "minor", "both", "none"
    unit : string
        Unit of frequency.
    heading : string
        Provide a heading for the image. If None, no heading.
    figtxt : string (multiline possible)
        Provide an annotation for a box below the figure. If None, no annotaion.

    Yields
    ------
    One saved image in path.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    font = {"family": "DejaVu Sans", "weight": "normal", "size": 20}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=15)
    plt.figure(figsize=[20, 10], dpi=300)
    if np.ndim(data) == 1:
        plt.loglog(
            frequency,
            data,
            label=str(labels[0]),
            linewidth=1,
            linestyle=linestyle,
            marker=marker,
            #markersize=markersize,
        )
    elif (np.ndim(data) != 1) & (np.ndim(frequency) == 1):
        for i, spectrum in enumerate(data):
            plt.loglog(
                frequency,
                spectrum,
                label=labels[i],
                linewidth=1,
                linestyle=linestyle[i],
                marker=marker[i],
                #markersize=markersize[i],
            )
    else:
        for i, spectrum in enumerate(data):
            plt.loglog(
                frequency[i],
                spectrum,
                label=labels[i],
                linewidth=1,
                linestyle=linestyle[i],
                marker=marker[i],
                #markersize=markersize[i],
            )
    plt.grid(which=grid, color="grey", linestyle="-", linewidth=0.2)
    if lims != None:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    if heading != None:
        plt.title(heading)
    if labels != None:
        plt.ylabel("Spectral Density")
        plt.xlabel("Frequency %s" % unit)
    plt.legend(loc="best")
    if figtxt != None:
        plt.figtext(
            0.135,
            -0.05,
            figtxt,
            horizontalalignment="left",
            bbox=dict(boxstyle="square", facecolor="#F2F3F4", ec="1", pad=0.8, alpha=1),
        )
    if path != None:
        import datetime

        if name == None:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(
            path + "/" + comment + name + ".png", pad_inches=0.5, bbox_inches="tight"
        )
    else:
        plt.show()
    plt.close()
