# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional, Union

import pandas as pd
import numpy as np
from pandas import DataFrame
from IPython.display import display, HTML

from ..graph import ScatterGraph, BarGraph
from ..utils import guess_plotly_rangebreaks


def _calculate_group_returns(pred_label: pd.DataFrame, group_num: int, factor_direction: int, label_period: int =72) -> tuple:
    """
    Calculate group returns and related metrics.
    
    Parameters:
        pred_label (pd.DataFrame): MultiIndex DataFrame with columns 'factor', 'label_24', etc.
        group_num (int): Number of groups to split factors into.
        factor_direction (int): -1 or 1, indicating the direction of the factor.
    
    Returns:
        tuple: DataFrames of group returns, long-short returns, weighted returns, etc.  
    """
    # Ensure 'factor' is present
    assert 'factor' in pred_label.columns, "pred_label must contain 'factor' column."
    label_cols = [col for col in pred_label.columns if col.startswith('label')]
    assert len(label_cols) >= 1, "pred_label must contain at least one label column (e.g., 'label_24')."
    
    # For each label column, compute group returns
    all_results = {}
    for label_col in label_cols:
        df = pred_label[['factor', label_col]].copy()
        df = df.dropna()
        
        # Rank factor within each datetime
        df['factor_rank'] = df.groupby(level='datetime')['factor'].rank(method='average', ascending=(factor_direction == 1))
        
        # Assign group based on rank      TODO 解决分组失败的时候考虑是否应该直接丢弃，还是说需要应该最少分组数量
        df['group'] = df.groupby(level='datetime')['factor_rank'].transform(
            lambda x: pd.qcut(x, q=group_num, labels=False, duplicates='drop') + 1
        )

        # Calculate group returns (equal-weighted)
        group_returns = df.groupby([pd.Grouper(level='datetime'), 'group'])[label_col].mean().unstack()
        group_returns.columns = [f'Group {g}' for g in group_returns.columns]
        group_returns = group_returns.iloc[::label_period] # Downsample to label period
        group_returns = group_returns.dropna() # Drop missing values

        # Long-short returns (Group 1 vs Group N)
        # 统一把 group 列名中的数字从 float 转为 int（例如 'Group 1.0' -> 'Group 1'）
        group_returns.columns = [f'Group {int(float(g.split()[1]))}' for g in group_returns.columns]
        long_short_returns = group_returns[f'Group {group_num}'] - group_returns['Group 1']
        long_short_returns.name = 'long_short_return'
        
        # Weighted group returns (linear weights from middle to ends) TODO 分组失败的时候该权重矩阵计算是否会出错
        mid = (group_num + 1) / 2
        offsets = np.arange(1, group_num + 1) - mid  # 生成对称偏移量（如group_num=5时：[-2,-1,0,1,2]
        max_offset = np.max(np.abs(offsets))  # 获取最大偏移量
        target_max = 0.2  # 目标最大绝对值（可根据需求调整）
        weights = offsets * (target_max / max_offset)  # 归一化到对称线性分布（如[-0.2,-0.1,0,0.1,0.2]）
        weighted_returns = (group_returns * weights).sum(axis=1)
        weighted_returns.name = 'weighted_return'
        
        # Monthly returns for long-short (consider label period)
        # Filter returns to match label period (e.g., 72-hour intervals)
        # monthly_returns = long_short_returns.resample('M').sum() # 使用首尾的对冲收益率
        monthly_returns = weighted_returns.resample('M').sum() # 使用加权收益率
        monthly_returns.index = monthly_returns.index.astype(str)
        # monthly_win_rate = (long_short_returns > 0).groupby(long_short_returns.index.to_period('M')).mean()  # 计算每月内long_short_returns大于0的概率
        monthly_win_rate = (weighted_returns > 0).groupby(weighted_returns.index.to_period('M')).mean()  # 计算每月内weighted_returns大于0的概率
        monthly_win_rate.index = monthly_win_rate.index.astype(str)  # 或 to_timestamp()

        # Autocorrelation of factor ranks
        valid_dates = sorted(set(df.index.get_level_values('datetime')) & set(group_returns.index))
        factor_matrix = df['factor_rank'].unstack(level='datetime')[valid_dates]
        factor_matrix = factor_matrix.sort_index()
        autocorrs = [
            factor_matrix[valid_dates[i]].corr(factor_matrix[valid_dates[i - 1]])
            for i in range(1, len(valid_dates))
        ]
        factor_rank_autocorr = pd.Series(autocorrs, index=valid_dates[1:], name='factor_rank_autocorr')

        # TODO 实现换手率逻辑
        # Calculate turnover based on long-short group instrument changes
        # Filter long (group=1) and short (group=group_num) instruments at each datetime
        # Filter long-short data to only include datetimes with valid group numbers (from group_returns index)
        long_short = df[df['group'].isin([1, group_num])].reset_index()
        long_short = long_short[long_short['datetime'].isin(group_returns.index)]

        # Get instrument sets for each rebalance period
        period_instruments = long_short.groupby('datetime')['instrument'].apply(set)
        
        # 计算相邻时间点的变动数量
        turnover = period_instruments.diff().apply(
            lambda x: len(x) if isinstance(x, set) else 0
        )
        # 分母为当前期 instrument 数量（用于归一化）
        base_count = period_instruments.apply(len)
        # 换手率 = 变化数 / 当前数量
        turnover_rate = turnover.div(base_count)
        turnover_rate.name = 'turnover'
        # 年化换手率（按 rebalance 次数估算）
        annual_periods = (365 * 24) / label_period
        turnover_annualized = turnover_rate.rolling(window=12, min_periods=1).mean() * annual_periods
        turnover_annualized.name = 'annualized_turnover'
        
        all_results[label_col] = {
            'group_returns': group_returns,
            'long_short_returns': long_short_returns,
            'weighted_returns': weighted_returns,
            'monthly_returns': monthly_returns,
            'monthly_win_rate': monthly_win_rate,
            'factor_rank_autocorr': factor_rank_autocorr,
            'turnover_annualized': turnover_annualized
        }
    
    return all_results


def _compute_metrics(returns: pd.Series) -> pd.DataFrame:
    """
    Compute performance metrics: Sharpe ratio, max drawdown, Calmar ratio.
    
    Parameters:
        returns (pd.Series): Periodic returns.
    
    Returns:
        pd.DataFrame: Metrics summary.
    """
    annualized_return = (1 + returns).prod() ** (365 / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(365)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
    
    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding().max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = annualized_return / (-max_drawdown) if max_drawdown != 0 else np.nan
    
    return pd.DataFrame({
        'Annualized Sharpe Ratio': [sharpe_ratio],
        'Max Drawdown': [max_drawdown],
        'Calmar Ratio': [calmar_ratio]
    }, index=['Metrics'])


def factors_group_test_graph(pred_label: pd.DataFrame, group_num: int = 10, factor_direction: int = 1,
                             show_notebook: bool = True, label_period: int = 72, **kwargs) -> Union[DataFrame, list[DataFrame]]:
    """
    Generate visualizations and metrics for factor group testing.
    
    Parameters:
        pred_label (pd.DataFrame): MultiIndex DataFrame with index ['datetime', 'instrument'], columns ['factor', 'label_24', ...].
        group_num (int): Number of groups to split factors into. Default 10.
        factor_direction (int): -1 or 1, indicating factor direction. Default 1.
        show_notebook (bool): Whether to display in notebook. Default True.
        label_period (int): Label period in hours. Default 72.
    
    Returns:
        list or tuple: Plotly figures if show_notebook is False.
    """
    # Validate inputs
    assert factor_direction in [-1, 1], "factor_direction must be -1 or 1."
    assert group_num >= 2, "group_num must be at least 2."
    
    # Calculate group returns and metrics
    results = _calculate_group_returns(pred_label, group_num, factor_direction, label_period)
    figures = []
    
    # Generate visualizations for each label column
    for label_col, data in results.items():
        # Group净值走势图
        group_cum_returns = (1 + data['group_returns']).cumprod()
        fig_group = ScatterGraph(
            group_cum_returns,
            layout=dict(title=f'{label_col} Group Cumulative Returns', xaxis=dict(tickangle=45, rangebreaks=guess_plotly_rangebreaks(group_cum_returns.index))),
            graph_kwargs={'mode': 'lines'}
        ).figure
        figures.append(fig_group)
        
        # Long-short净值走势图
        ls_cum_returns = pd.DataFrame({'Long-Short Cumulative Return': (1 + data['long_short_returns']).cumprod()})
        fig_ls = ScatterGraph(
            ls_cum_returns,
            layout=dict(title=f'{label_col} Long-Short Cumulative Returns', xaxis=dict(tickangle=45, rangebreaks=guess_plotly_rangebreaks(ls_cum_returns.index))),
            graph_kwargs={'mode': 'lines'}
        ).figure
        figures.append(fig_ls)
        
        # Weighted组合净值走势图
        weighted_cum_returns = (1 + data['weighted_returns']).cumprod().to_frame(name='weighted_cum_returns')  # 转换为DataFrame以满足ScatterGraph需求
        fig_weighted = ScatterGraph(
            weighted_cum_returns,
            layout=dict(title=f'{label_col} Weighted Cumulative Returns', xaxis=dict(tickangle=45, rangebreaks=guess_plotly_rangebreaks(weighted_cum_returns.index))),
            graph_kwargs={'mode': 'lines'}
        ).figure
        figures.append(fig_weighted)
        
        # Monthly胜率柱状图
        monthly_win_rate = data['monthly_win_rate'].to_frame(name='monthly_win_rate')
        fig_win = BarGraph(
            monthly_win_rate,
            layout=dict(title=f'{label_col} Monthly Win Rate', xaxis=dict(tickangle=45), yaxis=dict(title='Win Rate')),
            graph_kwargs={'type': 'bar'}
        ).figure
        figures.append(fig_win)
        
        # Monthly收益柱状图
        monthly_returns = data['monthly_returns'].to_frame(name='monthly_returns')
        fig_monthly = BarGraph(
            monthly_returns,
            layout=dict(title=f'{label_col} Monthly Returns', xaxis=dict(tickangle=45), yaxis=dict(title='Return')),
            graph_kwargs={'type': 'bar'}
        ).figure
        figures.append(fig_monthly)
        
        # 每组平均收益率柱状图
        avg_group_returns = data['group_returns'].mean().to_frame(name='group_returns')
        fig_avg = BarGraph(
            avg_group_returns,
            layout=dict(title=f'{label_col} Average Group Returns', xaxis=dict(title='Group'), yaxis=dict(title='Average Return')),
            graph_kwargs={'type': 'bar'}
        ).figure
        figures.append(fig_avg)
        
        # Factor rank自相关时序图
        autocorr = data['factor_rank_autocorr'].to_frame(name='factor_rank_autocorr')
        fig_autocorr = ScatterGraph(
            autocorr,
            layout=dict(title=f'{label_col} Factor Rank Autocorrelation', xaxis=dict(tickangle=45, rangebreaks=guess_plotly_rangebreaks(autocorr.index))),
            graph_kwargs={'mode': 'lines+markers'}
        ).figure
        figures.append(fig_autocorr)
        
        # 年化换手率时序图
        turnover = data['turnover_annualized'].to_frame(name='turnover_annualized')
        fig_turnover = ScatterGraph(
            turnover,
            layout=dict(title=f'{label_col} Annualized Turnover', xaxis=dict(tickangle=45, rangebreaks=guess_plotly_rangebreaks(turnover.index))),
            graph_kwargs={'mode': 'lines+markers'}
        ).figure
        figures.append(fig_turnover)
    
    # Compute and display metrics
    metrics_dfs = []
    for label_col, data in results.items():
        # 计算基础指标（原long_short_returns相关）
        # base_metrics = _compute_metrics(data['long_short_returns']) # 使用首尾多空组合收益率计算夏普比
        base_metrics = _compute_metrics(data['weighted_returns'])

        # 新增指标：monthly win rate（月胜率）平均值
        monthly_win_rate_mean = data['monthly_win_rate'].mean()
        
        # 新增指标：monthly returns（月收益率）平均值
        monthly_returns_mean = data['monthly_returns'].mean()
        
        # 新增指标：factor rank Autocorrelation（因子秩自相关）平均值
        factor_autocorr_mean = data['factor_rank_autocorr'].mean()

        # 构造额外的指标作为 DataFrame，和 base_metrics 对齐列拼接
        extra_metrics = pd.DataFrame({
            'monthly_win_rate_mean': [monthly_win_rate_mean],
            'monthly_returns_mean': [monthly_returns_mean],
            'factor_autocorr_mean': [factor_autocorr_mean]
        }, index=base_metrics.index)  # 保持相同行索引（比如 "Metrics"）

        # 列方向拼接
        metrics = pd.concat([base_metrics, extra_metrics], axis=1)
        # 转换为DataFrame行，行索引为label_col
        # 将Series转换为单行DataFrame（行索引为指标名，列索引为默认），再转置为1行多列结构
        metrics_df = metrics
        metrics_df.index = [label_col]
        metrics_dfs.append(metrics_df)
    metrics_summary = pd.concat(metrics_dfs)

    if show_notebook:
        # print("Metrics Summary:\n", metrics_summary)
        print("Metrics Summary:")
        display(metrics_summary)  # 关键点：使用 display() 来触发富格式渲染
        ScatterGraph.show_graph_in_notebook(figures)
        return metrics_summary
    else:
        return figures + [metrics_summary]