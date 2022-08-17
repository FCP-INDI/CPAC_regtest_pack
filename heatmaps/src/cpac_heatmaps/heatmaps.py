"""Visualize correlations between runs

Copyright (C) 2022  C-PAC Developers
This file is part of CPAC_regtest_pack.
CPAC_regtest_pack is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
CPAC_regtest_pack is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with CPAC_regtest_pack. If not, see
<https://www.gnu.org/licenses/>"""
import argparse
from warnings import filterwarnings
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

filterwarnings("ignore", "Warning: converting a masked element to nan")


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=None, threshold=None, **textkw):
    """Annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if textcolors is None:
        textcolors = ["black", "white"]

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None),
                                fontsize=5 if '\n' in data[i, j] else 10,
                                **kw)
            texts.append(text)

    return texts


def generate_heatmap(corrs, var_list, sub_list, save_path=None, title=None):
    """Generate a heatmap.

    Parameters
    ----------
    corrs: numpy ndarray with shape (number of features, number of unique IDs)
        This matrix contains the values to plot
    var_list: list of strings
        The labels, in order, of the features (rows)
    sub_list: list of strings
        The labels, in order, of the subject_sessions (columns)
    save_path: string or falsy
        The path to save the file to, or a falsy value to display in IPython
    title: str
        String to use as plot title. Optional.

    Returns
    -------
    None
    """
    def figsize(shape):
        if isinstance(shape, tuple):
            return tuple(figsize(dim) for dim in shape)
        if shape < 5:
            return shape * 5
        if shape < 10:
            return shape * 2
        return shape
    if not isinstance(corrs, np.ndarray):
        corrs = np.array(corrs)
    corrs = corrs[~np.isnan(corrs).all(axis=1)]  # drop all-NaN rows
    corrs = corrs[:, ~np.isnan(corrs).all(axis=0)]  # drop all-NaN columns
    fig, ax = plt.subplots(figsize=(figsize(corrs.shape)))
    im, _ = heatmap(
        corrs, var_list, sub_list, ax=ax, vmin=0, vmax=1,
        cbarlabel="correlation score"
    )
    annotate_heatmap(im)
    if title:
        plt.title(
            label=title,
            fontdict={
                'fontsize': max(24, len(sub_list)*0.75),
                'fontweight' : 'bold',
            }
        )
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        try:
            from IPython.display import display
            plt.show()
        except:
            print("No save path or display configured")


# pylint: disable=too-many-arguments
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbar_kw is None:
        cbar_kw = {}
    cbar = ax.figure.colorbar(
        im, ax=ax, values=None, boundaries=None, fraction=0.03, pad=0.03,
        **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, fontsize=20, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), fontsize=10, rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for _, spine in list(ax.spines.items()):
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="generate heatmaps"
    )

    parser.add_argument(
        'config',
        help='path to a YAML configuration file specifying the data, '
             'features and participants to plot'
    )

    parser.add_argument(
        '-o', '--output',
        dest='save_path',
        help='path to save heatmap to',
        required=False
    )

    parsed = vars(parser.parse_args(args[1:] if len(args)>1 else args))
    return(parsed.pop('config'), parsed)