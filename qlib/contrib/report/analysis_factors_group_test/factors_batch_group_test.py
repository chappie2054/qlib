# factors_batch_group_test.py

import numpy as np
import pandas as pd
from typing import Tuple
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from ..graph import ScatterGraph, BarGraph
from ..utils import guess_plotly_rangebreaks

def _rankic_direction(factor: pd.Series, label: pd.Series) -> int:
    """
    计算因子与label的RankIC，用于判断因子方向。
    返回1或-1。
    """
    # 对每个时间截面计算秩相关
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


def _vectorized_group_test(factor: pd.Series, label: pd.Series, group_num: int, direction: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    使用向量化实现高效分组收益率和加权收益率的计算。
    返回：
        - group_returns_df: 每组平均收益率
        - group_cum_returns_df: 每组累计收益率（百分比形式）
    """
    # 拆解index
    datetimes = factor.index.get_level_values(0)
    instruments = factor.index.get_level_values(1)
    
    # 排序索引并整理成矩阵
    df = pd.DataFrame({
        'datetime': datetimes,
        'instrument': instruments,
        'factor': factor.values,
        'label': label.values
    }).dropna()
    
    df['factor'] = df['factor'] * direction

    # 获取唯一时间点
    unique_times = np.sort(df['datetime'].unique())
    time_map = {t: i for i, t in enumerate(unique_times)}
    df['time_idx'] = df['datetime'].map(time_map)

    # 编码 instrument
    unique_instruments = np.sort(df['instrument'].unique())
    inst_map = {inst: i for i, inst in enumerate(unique_instruments)}
    df['inst_idx'] = df['instrument'].map(inst_map)

    T = len(unique_times)
    N = len(unique_instruments)

    # 初始化矩阵
    factor_matrix = np.full((T, N), np.nan, dtype=np.float32)
    label_matrix = np.full((T, N), np.nan, dtype=np.float32)

    factor_matrix[df['time_idx'], df['inst_idx']] = df['factor']
    label_matrix[df['time_idx'], df['inst_idx']] = df['label']

    # 对每个时间进行分组
    sort_idx = np.argsort(factor_matrix, axis=1)
    sorted_label = np.take_along_axis(label_matrix, sort_idx, axis=1)

    valid_counts = np.sum(~np.isnan(sorted_label), axis=1)
    group_size = (valid_counts[:, None] / group_num).astype(int)

    # 构造掩码矩阵，标记每个位置所属分组
    group_returns = np.zeros((T, group_num), dtype=np.float32)
    for t in range(T):
        count = valid_counts[t]
        if count < group_num:
            group_returns[t, :] = np.nan
            continue
        labels_t = sorted_label[t, :count]
        sizes = [count // group_num + (1 if i < count % group_num else 0) for i in range(group_num)]
        start = 0
        for g in range(group_num):
            end = start + sizes[g]
            group_returns[t, g] = np.mean(labels_t[start:end]) if end > start else np.nan
            start = end

    # 加权多空组合收益
    mid = (group_num + 1) / 2
    offsets = np.arange(1, group_num + 1) - mid
    max_offset = np.max(np.abs(offsets))
    target_max = 0.2
    weights = offsets * (target_max / max_offset)
    weighted_returns = np.nansum(group_returns * weights[None, :], axis=1)

    # 构造时间索引
    dt_index = pd.to_datetime(unique_times)

    group_returns_df = pd.DataFrame(group_returns, columns=[f'Group {i + 1}' for i in range(group_num)], index=dt_index)
    group_cum_returns_df = (group_returns_df.cumsum() * 100).fillna(0)  # 百分比累计收益率
    weighted_returns_series = pd.Series(weighted_returns, index=dt_index, name='Weighted Return')
    weighted_returns_df = (weighted_returns_series.cumsum() * 100).to_frame().fillna(0)

    return group_returns_df, group_cum_returns_df, weighted_returns_df


def _process_single_factor(col, factor_series, label_series, group_num):
    direction = _rankic_direction(factor_series, label_series)
    group_returns_df, group_cum_returns_df, weighted_returns_df = _vectorized_group_test(
        factor_series, label_series, group_num, direction
    )
    avg_returns = group_returns_df.mean()
    avg_returns.name = col[1]
    return col[1], avg_returns, group_returns_df, group_cum_returns_df, weighted_returns_df


def batch_factors_group_test(factors_df: pd.DataFrame, group_num: int = 5, show_notebook: bool = True):
    assert isinstance(factors_df.columns, pd.MultiIndex)

    # 提取 label 列
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

    # 特征列表
    feature_cols = [col for col in factors_df.columns if col[0] == 'feature']

    # 并行处理每个因子
    results = Parallel(n_jobs=-1)(
        delayed(_process_single_factor)(
            col,
            sampled_factors_df[col].astype(np.float32),
            sampled_label_series,
            group_num
        )
        for col in tqdm(feature_cols, desc=f'Processing Factors (period={rebalance_period})')
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
                            xaxis=dict(tickangle=45, rangebreaks=guess_plotly_rangebreaks(weighted_returns_df.index))),
                graph_kwargs={'mode': 'lines'}
            ).figure

            all_figs.extend([fig_avg, fig_cum, fig_weighted])

    result_df = pd.DataFrame(all_avg_returns)

    if show_notebook:
        ScatterGraph.show_graph_in_notebook(all_figs)
    return result_df

