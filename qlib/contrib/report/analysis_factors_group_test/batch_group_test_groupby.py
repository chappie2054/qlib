import numpy as np
import pandas as pd
from typing import Tuple
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from ..graph import ScatterGraph, BarGraph
from ..utils import guess_plotly_rangebreaks, _rankic_direction


def _group_test_groupby(factor: pd.Series, label: pd.Series, group_num: int, direction: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    使用 groupby + 分组方式实现因子分组测试。
    返回：
        - group_returns_df: 每组平均收益率
        - group_cum_returns_df: 每组累计收益率
        - weighted_returns_df: 加权多空组合累计收益
    """
    factor = factor * direction
    df = pd.DataFrame({'factor': factor, 'label': label}).dropna()
    df = df.reset_index()  # ✅ 展开 MultiIndex，避免和列名冲突

    group_returns = {}
    weighted_returns = {}

    for dt, group in df.groupby('datetime'):
        if group.shape[0] < group_num:
            continue
        group = group.copy()
        group['group'] = pd.qcut(group['factor'].rank(method='first'), group_num, labels=False)
        returns = group.groupby('group')['label'].mean()
        if len(returns) < group_num:
            continue
        group_returns[dt] = returns.values

        # 加权收益（两端权重大，中间权重小）
        mid = (group_num + 1) / 2
        offsets = np.arange(1, group_num + 1) - mid
        weights = offsets * (0.2 / np.max(np.abs(offsets)))
        weighted_returns[dt] = np.dot(returns.values, weights)

    group_returns_df = pd.DataFrame(group_returns).T.sort_index()
    group_returns_df.columns = [f'Group {i+1}' for i in range(group_num)]
    group_cum_returns_df = group_returns_df.cumsum() * 100
    weighted_returns_df = pd.Series(weighted_returns).sort_index().cumsum() * 100
    weighted_returns_df = weighted_returns_df.to_frame(name='Weighted Return')

    return group_returns_df, group_cum_returns_df, weighted_returns_df


def _process_single_factor(col, factor_series, label_series, group_num):
    direction = _rankic_direction(factor_series, label_series)
    group_returns_df, group_cum_returns_df, weighted_returns_df = _group_test_groupby(
        factor_series, label_series, group_num, direction
    )
    avg_returns = group_returns_df.mean()
    avg_returns.name = col[1]
    return col[1], avg_returns, group_returns_df, group_cum_returns_df, weighted_returns_df


def batch_factors_group_test(factors_df: pd.DataFrame, group_num: int = 10, show_notebook: bool = True):
    assert isinstance(factors_df.columns, pd.MultiIndex)
    label_col = [col for col in factors_df.columns if col[0] == 'label'][0]
    label_series = factors_df[label_col].astype(np.float32)

    # 提取调仓周期（假设形如 'label_72'）
    try:
        rebalance_period = int(label_col[1].split('_')[1])
    except (IndexError, ValueError):
        raise ValueError(f"label 名 {label_col[1]} 不是合法格式，无法解析周期数")

    # 获取采样的时间点（每rebalance_period取一个）
    unique_times = sorted(label_series.index.get_level_values(0).unique())
    selected_times = unique_times[::rebalance_period]

    # 只保留调仓时间点上的数据
    sampled_index = factors_df.index.get_level_values(0).isin(selected_times)
    sampled_factors_df = factors_df.loc[sampled_index]
    sampled_label_series = label_series.loc[sampled_index]

    feature_cols = [col for col in factors_df.columns if col[0] == 'feature']

    results = Parallel(n_jobs=-1)(
        delayed(_process_single_factor)(
            col, sampled_factors_df[col].astype(np.float32), sampled_label_series, group_num
        )
        for col in tqdm(feature_cols, desc='Processing Factors')
    )

    all_avg_returns = []
    all_figs = []

    for colname, avg_returns, group_returns_df, group_cum_returns_df, weighted_returns_df in results:
        all_avg_returns.append(avg_returns)
        if show_notebook:
            avg_returns_df = avg_returns.to_frame(name='avg_returns')
            fig_avg = BarGraph(
                avg_returns_df,
                layout=dict(title=f'{colname} Average Group Returns', xaxis=dict(title='Group'),
                            yaxis=dict(title='Return')),
                graph_kwargs={'type': 'bar'}
            ).figure
            fig_cum = ScatterGraph(
                group_cum_returns_df,
                layout=dict(title=f'{colname} Cumulative Group Returns (%)',
                            xaxis=dict(tickangle=45,
                                       rangebreaks=guess_plotly_rangebreaks(group_cum_returns_df.index))),
                graph_kwargs={'mode': 'lines'}
            ).figure
            fig_weighted = ScatterGraph(
                weighted_returns_df,
                layout=dict(title=f'{colname} Weighted Return (%)',
                            xaxis=dict(tickangle=45,
                                       rangebreaks=guess_plotly_rangebreaks(weighted_returns_df.index))),
                graph_kwargs={'mode': 'lines'}
            ).figure
            all_figs.extend([fig_avg, fig_cum, fig_weighted])

    result_df = pd.DataFrame(all_avg_returns)
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(all_figs)
    return result_df
