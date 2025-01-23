import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from utils import gpu2cpu

try:
    import cupy as cp
except ImportError as error:
    print("No cupy, using numpy instead")
    import numpy as cp


def phase_plot(z: np.ndarray, gamma: float = 1) -> np.ndarray:
    z = gpu2cpu(z)
    z[np.isinf(z)] = np.max(abs(np.ma.array(z, mask=~np.isfinite(z))))
    z[np.isnan(z)] = 0
    amp = np.atleast_3d(abs(z) / np.max(abs(z))) ** gamma + 1e-16
    phase = np.angle(z)
    colors = np.stack(
        ((np.sin(phase) + 1), (np.cos(phase) + 1), (np.sin(-phase) + 1)), axis=2
    )
    return amp * colors / 2


def plot_rec_lines(
    rec_errs, meas_errs, plot_path=None, ax: plt.Axes = None, save=False, title_str=""
):
    rec_errs, meas_errs = gpu2cpu([rec_errs, meas_errs])
    # plt.style.use(("science", "no-latex"))
    cb = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=200)
    inds = range(5, rec_errs.shape[1])
    for i in range(rec_errs.shape[0]):
        ax.plot(inds, 10 * np.log10(rec_errs[i, inds]), alpha=0.2, color=cb[0], linewidth=0.5)
        ax.plot(inds, 10 * np.log10(meas_errs[i, inds]), alpha=0.2, color=cb[1], linewidth=0.5)
    # ax.set_xlim([0, meas_errs.argmin(axis=1).max()])
    # print(meas_errs.min(axis=1).argmax())
    ax.set_ylim([-34, 4])
    ax.set_xlim([0, 4e3])
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.minorticks_off()
    ax.legend(
        [Line2D([], [], color=cb[0]), Line2D([], [], color=cb[1])],
        ["Field", "Measurement"],
    )
    ax.set_title("Reconstruction error" + title_str)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("NMSE [dB]")
    if save is True:
        plt.savefig(plot_path)


def scatter_rec_rec(
    x, y, ax: plt.Axes = None, save: bool = False, plot_path: str = None, color_ind=0,
):
    x[x == 0] = np.inf
    y[y == 0] = np.inf
    # plt.style.use(("science", "no-latex"))
    cb_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    x, y = gpu2cpu([x, y])
    ax.scatter(
        10 * np.log10(x.min(axis=1)),
        10 * np.log10(y.min(axis=1)),
        s=16,
        linewidths=.5,
        edgecolors=cb_colors[color_ind] + "44",
        facecolor=[1, 1, 1, 0],
        # alpha=0.8,
        marker='o'
    )
    if save is True:
        plt.savefig(plot_path)
    return ax
