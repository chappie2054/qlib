# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def sub_fig_generator(sub_figsize=(3, 3), col_n=10, row_n=1, wspace=None, hspace=None, sharex=False, sharey=False):
    """sub_fig_generator.
    it will return a generator, each row contains <col_n> sub graph

    FIXME: Known limitation:
    - The last row will not be plotted automatically, please plot it outside the function

    Parameters
    ----------
    sub_figsize :
        the figure size of each subgraph in <col_n> * <row_n> subgraphs
    col_n :
        the number of subgraph in each row;  It will generating a new graph after generating <col_n> of subgraphs.
    row_n :
        the number of subgraph in each column
    wspace :
        the width of the space for subgraphs in each row
    hspace :
        the height of blank space for subgraphs in each column
        You can try 0.3 if you feel it is too crowded

    Returns
    -------
    It will return graphs with the shape of <col_n> each iter (it is squeezed).
    """
    assert col_n > 1

    while True:
        fig, axes = plt.subplots(
            row_n, col_n, figsize=(sub_figsize[0] * col_n, sub_figsize[1] * row_n), sharex=sharex, sharey=sharey
        )
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        axes = axes.reshape(row_n, col_n)

        for col in range(col_n):
            res = axes[:, col].squeeze()
            if res.size == 1:
                res = res.item()
            yield res
        plt.show()


def guess_plotly_rangebreaks(dt_index: pd.DatetimeIndex):
    """
    This function `guesses` the rangebreaks required to remove gaps in datetime index.
    It basically calculates the difference between a `continuous` datetime index and index given.

    For more details on `rangebreaks` params in plotly, see
    https://plotly.com/python/reference/layout/xaxis/#layout-xaxis-rangebreaks

    Parameters
    ----------
    dt_index: pd.DatetimeIndex
    The datetimes of the data.

    Returns
    -------
    the `rangebreaks` to be passed into plotly axis.

    """
    dt_idx = dt_index.sort_values()
    gaps = dt_idx[1:] - dt_idx[:-1]
    min_gap = gaps.min()
    gaps_to_break = {}
    for gap, d in zip(gaps, dt_idx[:-1]):
        if gap > min_gap:
            gaps_to_break.setdefault(gap - min_gap, []).append(d + min_gap)
    return [dict(values=v, dvalue=int(k.total_seconds() * 1000)) for k, v in gaps_to_break.items()]


def _rankic_direction(factor: pd.Series, label: pd.Series) -> int:
    """
    计算因子与label的RankIC，用于判断因子方向。
    返回1或-1。
    """
    dates = factor.index.get_level_values(0).unique()
    rankics = []
    for dt in dates:
        try:
            fac_slice = factor.loc[dt]
            lab_slice = label.loc[dt]
            common_idx = fac_slice.dropna().index.intersection(lab_slice.dropna().index)
            if len(common_idx) >= 5:
                fac_rank = fac_slice.loc[common_idx].rank()
                lab_rank = lab_slice.loc[common_idx].rank()
                rankic = fac_rank.corr(lab_rank)
                if not np.isnan(rankic):
                    rankics.append(rankic)
        except Exception:
            continue
    mean_rankic = np.nanmean(rankics)
    return 1 if mean_rankic >= 0 else -1
