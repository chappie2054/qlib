# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Union
from tqdm.notebook import tqdm


def _rank_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    对Tensor的最后一维进行排序，返回百分比排名

    Parameters:
        x (torch.Tensor): 输入张量

    Returns:
        torch.Tensor: 百分比排名张量
    """
    # argsort两次是获取排名的标准技巧
    ranks = x.argsort(dim=-1).argsort(dim=-1)
    # 转换为百分比排名 (从0开始)
    return ranks.float() / (x.shape[-1] - 1)


def pearson_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    计算皮尔逊相关系数
    
    Parameters:
        x (torch.Tensor): 第一个张量
        y (torch.Tensor): 第二个张量
        dim (int): 计算维度，默认为-1
        eps (float): 防止除零的小常数
        
    Returns:
        torch.Tensor: 皮尔逊相关系数
    """
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    
    vx = x - x_mean
    vy = y - y_mean
    
    corr = torch.sum(vx * vy, dim=dim) / (torch.sqrt(torch.sum(vx ** 2, dim=dim)) * torch.sqrt(torch.sum(vy ** 2, dim=dim)) + eps)
    return corr


def spearman_corr(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    计算斯皮尔曼等级相关系数
    
    Parameters:
        x (torch.Tensor): 第一个张量
        y (torch.Tensor): 第二个张量
        dim (int): 计算维度，默认为-1
        eps (float): 防止除零的小常数
        
    Returns:
        torch.Tensor: 斯皮尔曼等级相关系数
    """
    # 先对数据进行排序
    x_rank = _rank_tensor(x)
    y_rank = _rank_tensor(y)
    
    # 然后计算排序后的皮尔逊相关系数
    return pearson_corr(x_rank, y_rank, dim=dim, eps=eps)


def calculate_ic_metrics_torch(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用PyTorch高效地计算所有因子的IC, Rank IC, ICIR, Rank ICIR, IC T-Value, Rank IC T-Value, IC符号稳定性和Rank IC符号稳定性。
    
    Parameters:
        factors_df (pd.DataFrame): 输入数据框，包含因子和标签列，使用MultiIndex列格式
        
    Returns:
        pd.DataFrame: 包含IC, Rank IC, ICIR, Rank ICIR, IC T-Value, Rank IC T-Value, IC符号稳定性和Rank IC符号稳定性指标的DataFrame
    """
    assert isinstance(factors_df.columns, pd.MultiIndex)
    
    # 获取标签列
    label_col = [col for col in factors_df.columns if col[0] == 'label'][0]
    label_series = factors_df[label_col].astype(np.float32)
    
    # 获取因子列
    factor_cols = [col for col in factors_df.columns if col[0] == 'feature']
    factor_names = [col[1] for col in factor_cols]
    
    if not factor_cols:
        raise ValueError("未找到任何因子列。请确保因子列以'feature'开头。")
    
    print(f"使用PyTorch开始计算 {len(factor_cols)} 个因子的IC指标...")
    
    # 设备选择 - 自动选择GPU（如果可用）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU计算: {device}")
    else:
        device = torch.device("cpu")
        print(f"使用CPU计算: {device}")
    
    # 数据准备 - 重建DataFrame以匹配原有函数期望的格式
    df = pd.DataFrame({
        'datetime': factors_df.index.get_level_values(0),
        'instrument': factors_df.index.get_level_values(1),
        'label': label_series.values
    })
    
    # 添加因子列
    for i, col in enumerate(factor_cols):
        df[f'factor_{i}'] = factors_df[col].astype(np.float32).values
    
    df_sorted = df.sort_values(by=['datetime', 'instrument']).reset_index(drop=True)
    
    factors_tensor = torch.tensor(df_sorted[[f'factor_{i}' for i in range(len(factor_cols))]].values, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(df_sorted['label'].values, dtype=torch.float32, device=device)
    
    # 获取每个日期的边界
    dates = df_sorted['datetime'].values
    unique_dates, counts = np.unique(dates, return_counts=True)
    group_boundaries = torch.tensor(counts, device=device).cumsum(0)
    
    daily_ic_list = []
    daily_rank_ic_list = []
    
    start_idx = 0
    for end_idx in tqdm(group_boundaries, desc='计算每日IC指标'):
        # 提取当前日期的数据
        factor_group = factors_tensor[start_idx:end_idx]
        label_group = label_tensor[start_idx:end_idx]
        
        # 计算IC (皮尔逊相关系数)
        daily_ic_list.append(pearson_corr(factor_group.T, label_group))
        
        # 计算Rank IC (斯皮尔曼等级相关系数)
        daily_rank_ic_list.append(spearman_corr(factor_group.T, label_group))
        
        start_idx = end_idx
    
    # 聚合计算最终指标
    daily_ic = torch.stack(daily_ic_list)
    daily_rank_ic = torch.stack(daily_rank_ic_list)
    
    # 计算均值和标准差
    ic_mean = daily_ic.mean(dim=0)
    ic_std = daily_ic.std(dim=0)
    icir = ic_mean / (ic_std + 1e-8)
    
    # 计算IC的t值 (t-value = IC均值 / IC的标准误差)
    ic_std_error = ic_std / torch.sqrt(torch.tensor(len(daily_ic), device=device, dtype=torch.float32))
    ic_tvalue = ic_mean / (ic_std_error + 1e-8)
    
    rank_ic_mean = daily_rank_ic.mean(dim=0)
    rank_ic_std = daily_rank_ic.std(dim=0)
    rank_icir = rank_ic_mean / (rank_ic_std + 1e-8)
    
    # 计算Rank IC的t值 (t-value = Rank IC均值 / Rank IC的标准误差)
    rank_ic_std_error = rank_ic_std / torch.sqrt(torch.tensor(len(daily_rank_ic), device=device, dtype=torch.float32))
    rank_ic_tvalue = rank_ic_mean / (rank_ic_std_error + 1e-8)
    
    # 计算IC符号稳定性 (单周期IC符号与长期平均IC符号一致的周期占比)
    ic_sign_consistency = (torch.sign(daily_ic) == torch.sign(ic_mean)).float().mean(dim=0)
    
    # 计算Rank IC符号稳定性 (单周期Rank IC符号与长期平均Rank IC符号一致的周期占比)
    rank_ic_sign_consistency = (torch.sign(daily_rank_ic) == torch.sign(rank_ic_mean)).float().mean(dim=0)
    
    # 组装成最终的DataFrame
    results_df = pd.DataFrame({
        'IC Mean': ic_mean.cpu().numpy(),
        'IC Std': ic_std.cpu().numpy(),
        'ICIR': icir.cpu().numpy(),
        'IC T-Value': ic_tvalue.cpu().numpy(),
        'IC Sign Stability': ic_sign_consistency.cpu().numpy(),
        'Rank IC Mean': rank_ic_mean.cpu().numpy(),
        'Rank IC Std': rank_ic_std.cpu().numpy(),
        'Rank ICIR': rank_icir.cpu().numpy(),
        'Rank IC T-Value': rank_ic_tvalue.cpu().numpy(),
        'Rank IC Sign Stability': rank_ic_sign_consistency.cpu().numpy()
    }, index=factor_names)
    
    print("IC指标计算完成。")
    
    return results_df


def calculate_ic_metrics_pandas(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用Pandas计算所有因子的IC, Rank IC, ICIR, Rank ICIR, IC T-Value, Rank IC T-Value, IC符号稳定性和Rank IC符号稳定性。
    
    这是一个纯Pandas实现，适用于不想使用PyTorch的场景。
    
    Parameters:
        factors_df (pd.DataFrame): 输入数据框，包含因子和标签列，使用MultiIndex列格式
        
    Returns:
        pd.DataFrame: 包含IC, Rank IC, ICIR, Rank ICIR, IC T-Value, Rank IC T-Value, IC符号稳定性和Rank IC符号稳定性指标的DataFrame
    """
    assert isinstance(factors_df.columns, pd.MultiIndex)
    
    # 获取标签列
    label_col = [col for col in factors_df.columns if col[0] == 'label'][0]
    label_series = factors_df[label_col].astype(np.float32)
    
    # 获取因子列
    factor_cols = [col for col in factors_df.columns if col[0] == 'feature']
    factor_names = [col[1] for col in factor_cols]
    
    if not factor_cols:
        raise ValueError("未找到任何因子列。请确保因子列以'feature'开头。")
    
    print(f"使用Pandas开始计算 {len(factor_cols)} 个因子的IC指标...")
    
    # 数据准备 - 重建DataFrame以匹配原有函数期望的格式
    df = pd.DataFrame({
        'datetime': factors_df.index.get_level_values(0),
        'instrument': factors_df.index.get_level_values(1),
        'label': label_series.values
    })
    
    # 添加因子列
    for i, col in enumerate(factor_cols):
        df[f'factor_{i}'] = factors_df[col].astype(np.float32).values
    
    df_sorted = df.sort_values(by=['datetime', 'instrument']).reset_index(drop=True)
    
    daily_ic_results = []
    daily_rank_ic_results = []
    
    # 按日期分组计算
    for date, group_data in tqdm(df_sorted.groupby('datetime'), desc='计算每日IC指标'):
        if len(group_data) < 3:  # 至少需要3个样本才能计算相关系数
            continue
            
        daily_ic_row = {}
        daily_rank_ic_row = {}
        
        for i in range(len(factor_cols)):
            factor_col = f'factor_{i}'
            # 计算IC (皮尔逊相关系数)
            ic_corr = group_data[factor_col].corr(group_data['label'])
            daily_ic_row[factor_names[i]] = ic_corr
            
            # 计算Rank IC (斯皮尔曼等级相关系数)
            rank_ic_corr = group_data[factor_col].rank().corr(group_data['label'].rank())
            daily_rank_ic_row[factor_names[i]] = rank_ic_corr
        
        daily_ic_results.append(daily_ic_row)
        daily_rank_ic_results.append(daily_rank_ic_row)
    
    # 转换为DataFrame
    daily_ic_df = pd.DataFrame(daily_ic_results)
    daily_rank_ic_df = pd.DataFrame(daily_rank_ic_results)
    
    # 计算最终指标
    results = []
    for factor_name in factor_names:
        ic_series = daily_ic_df[factor_name].dropna()
        rank_ic_series = daily_rank_ic_df[factor_name].dropna()
        
        if len(ic_series) > 0:
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / (ic_std + 1e-8) if ic_std != 0 else np.nan
            # 计算IC的t值 (t-value = IC均值 / IC的标准误差)
            ic_std_error = ic_std / np.sqrt(len(ic_series))
            ic_tvalue = ic_mean / (ic_std_error + 1e-8) if ic_std_error != 0 else np.nan
            # 计算IC符号稳定性 (单周期IC符号与长期平均IC符号一致的周期占比)
            ic_sign_consistency = np.mean(np.sign(ic_series) == np.sign(ic_mean))
        else:
            ic_mean = np.nan
            ic_std = np.nan
            icir = np.nan
            ic_tvalue = np.nan
            ic_sign_consistency = np.nan
        
        if len(rank_ic_series) > 0:
            rank_ic_mean = rank_ic_series.mean()
            rank_ic_std = rank_ic_series.std()
            rank_icir = rank_ic_mean / (rank_ic_std + 1e-8) if rank_ic_std != 0 else np.nan
            # 计算Rank IC的t值 (t-value = Rank IC均值 / Rank IC的标准误差)
            rank_ic_std_error = rank_ic_std / np.sqrt(len(rank_ic_series))
            rank_ic_tvalue = rank_ic_mean / (rank_ic_std_error + 1e-8) if rank_ic_std_error != 0 else np.nan
            # 计算Rank IC符号稳定性 (单周期Rank IC符号与长期平均Rank IC符号一致的周期占比)
            rank_ic_sign_consistency = np.mean(np.sign(rank_ic_series) == np.sign(rank_ic_mean))
        else:
            rank_ic_mean = np.nan
            rank_ic_std = np.nan
            rank_icir = np.nan
            rank_ic_tvalue = np.nan
            rank_ic_sign_consistency = np.nan
        
        results.append({
            'IC Mean': ic_mean,
            'IC Std': ic_std,
            'ICIR': icir,
            'IC T-Value': ic_tvalue,
            'IC Sign Stability': ic_sign_consistency,
            'Rank IC Mean': rank_ic_mean,
            'Rank IC Std': rank_ic_std,
            'Rank ICIR': rank_icir,
            'Rank IC T-Value': rank_ic_tvalue,
            'Rank IC Sign Stability': rank_ic_sign_consistency
        })
    
    results_df = pd.DataFrame(results, index=factor_names)
    
    print("IC指标计算完成。")
    
    return results_df


def batch_calculate_ic_metrics(factors_df: pd.DataFrame, method: str = 'torch') -> pd.DataFrame:
    """
    批量计算所有因子的IC指标，支持多标签列。
    
    Parameters:
        factors_df (pd.DataFrame): 输入数据框，包含因子和标签列，使用MultiIndex列格式
        method (str): 计算方法，'torch'或'pandas'，默认为'torch'
        
    Returns:
        pd.DataFrame: 包含所有标签列的IC指标（包括IC, Rank IC, ICIR, Rank ICIR, IC T-Value, Rank IC T-Value, IC符号稳定性和Rank IC符号稳定性）的DataFrame
    """
    assert isinstance(factors_df.columns, pd.MultiIndex)
    
    # 获取标签列
    label_cols = [col for col in factors_df.columns if col[0] == 'label']
    
    if not label_cols:
        raise ValueError("未找到任何标签列。请确保标签列以'label'开头。")
    
    all_results = []
    
    for label_col in tqdm(label_cols, desc='批量计算IC指标'):
        # 创建临时DataFrame，只包含当前标签列
        temp_df = factors_df.copy()
        # 移除其他标签列
        for col in label_cols:
            if col != label_col:
                temp_df = temp_df.drop(columns=[col])
        
        if method == "torch":
            ic_metrics = calculate_ic_metrics_torch(temp_df)
        elif method == "pandas":
            ic_metrics = calculate_ic_metrics_pandas(temp_df)
        else:
            raise ValueError(f"不支持的方法: {method}。请选择'torch'或'pandas'")
        
        # 添加标签列信息
        ic_metrics['label_col'] = label_col[1]  # 获取标签名
        ic_metrics = ic_metrics.reset_index().rename(columns={'index': 'factor'})
        
        all_results.append(ic_metrics)
    
    # 合并所有结果
    final_results = pd.concat(all_results, ignore_index=True)
    
    return final_results