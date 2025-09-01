import numpy as np
import pandas as pd
import math
from scipy.stats import spearmanr
import logging

logger = logging.getLogger("GP-Factor")

# basic elementwise ops (pandas handles alignment)
def op_add(x, y): return x + y
def op_sub(x, y): return x - y
def op_mul(x, y): return x * y
def op_div(x, y):
    # safe div: if denominator is 0 -> NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        out = x / y
    if isinstance(out, pd.Series):
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def op_abs(x): return x.abs()
def op_sqrt(x):
    return np.sqrt(x.abs())
def op_log(x):
    return np.log(x.abs() + 1e-9)
def op_inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = 1.0 / x
    if isinstance(out, pd.Series):
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

# rank: cross-sectional rank per date (percentile 0..1)
def op_rank(x):
    # x: pandas Series with MultiIndex
    return x.groupby(level=0).rank(pct=True)

# delay: shift by d days within each instrument
def op_delay(x, d):
    return x.groupby(level="instrument").shift(d)

# delta: x - delay(x,d)
def op_delta(x, d):
    return x - op_delay(x, d)

# rolling helpers: operate per instrument with window d
def _rolling_group_apply_vectorized(s, d, func_name):
    # s: Series with MultiIndex
    # returns Series aligned
    def apply_group(g):
        # g is Series indexed by datetime
        rolling_obj = g.rolling(window=d, min_periods=1)
        return getattr(rolling_obj, func_name)()
    
    return s.groupby(level="instrument", group_keys=False).apply(apply_group)

def op_ts_sum(x, d):
    return _rolling_group_apply_vectorized(x, d, "sum")

def op_ts_product(x, d):
    # 使用向量化操作替代自定义函数
    return x.groupby(level="instrument").apply(
        lambda g: g.rolling(window=d, min_periods=1).apply(
            lambda arr: np.prod(np.where(np.isnan(arr), 1.0, arr)), raw=True
        )
    )

def op_ts_stddev(x, d):
    return _rolling_group_apply_vectorized(x, d, "std")

def op_ts_min(x, d):
    return _rolling_group_apply_vectorized(x, d, "min")

def op_ts_max(x, d):
    return _rolling_group_apply_vectorized(x, d, "max")

def op_ts_argmin(x, d):
    # 向量化实现argmin
    def rolling_argmin(g):
        # 使用pandas内置的idxmin然后计算偏移
        rolled = g.rolling(window=d, min_periods=1)
        # 计算每个窗口中最小值的索引位置
        result = pd.Series(index=g.index, dtype=float)
        for i in range(len(g)):
            start = max(0, i - d + 1)
            window = g.iloc[start:i+1]
            if len(window.dropna()) < 1:
                result.iloc[i] = np.nan
            else:
                try:
                    # 计算相对位置
                    result.iloc[i] = len(window) - 1 - (window[::-1].idxmin() - window.index[0])
                except:
                    result.iloc[i] = np.nan
        return result
    
    return x.groupby(level="instrument", group_keys=False).apply(rolling_argmin)

def op_ts_argmax(x, d):
    # 向量化实现argmax
    def rolling_argmax(g):
        # 使用pandas内置的idxmax然后计算偏移
        rolled = g.rolling(window=d, min_periods=1)
        # 计算每个窗口中最大值的索引位置
        result = pd.Series(index=g.index, dtype=float)
        for i in range(len(g)):
            start = max(0, i - d + 1)
            window = g.iloc[start:i+1]
            if len(window.dropna()) < 1:
                result.iloc[i] = np.nan
            else:
                try:
                    # 计算相对位置
                    result.iloc[i] = len(window) - 1 - (window[::-1].idxmax() - window.index[0])
                except:
                    result.iloc[i] = np.nan
        return result
    
    return x.groupby(level="instrument", group_keys=False).apply(rolling_argmax)

def op_ts_rank(x, d):
    # 向量化实现ts_rank
    def rolling_rank(g):
        rolled = g.rolling(window=d, min_periods=1)
        # 对每个窗口计算当前值的百分位排名
        def calc_rank(window):
            if len(window.dropna()) < 2:
                return np.nan
            current_val = window.iloc[-1]
            valid_vals = window.dropna()
            return (valid_vals < current_val).sum() / (len(valid_vals) - 1) if len(valid_vals) > 1 else np.nan
        
        return rolled.apply(calc_rank, raw=True)
    
    return x.groupby(level="instrument", group_keys=False).apply(rolling_rank)

# rolling correlation/covariance between X and Y for each instrument (series inputs)
def op_correlation_vectorized(x, y, d):
    # 将两个Series合并为DataFrame
    df = pd.DataFrame({'x': x, 'y': y})
    
    # 定义滚动相关系数计算函数
    def rolling_corr(g):
        # 使用pandas内置的rolling.corr方法
        return g['x'].rolling(window=d, min_periods=2).corr(g['y'])
    
    # 对每个instrument组应用滚动相关计算
    result = df.groupby(level="instrument").apply(rolling_corr)
    
    # 调整索引以匹配原始输入
    if not result.empty:
        result = result.droplevel(0).reorder_levels(x.index.names)
        # 确保索引顺序一致
        result = result.sort_index()
    else:
        result = pd.Series(dtype=float)
    
    return result

def op_covariance_vectorized(x, y, d):
    df = pd.DataFrame({'x': x, 'y': y})
    
    def rolling_cov(g):
        return g['x'].rolling(window=d, min_periods=2).cov(g['y'])
    
    result = df.groupby(level="instrument").apply(rolling_cov)
    
    if not result.empty:
        result = result.droplevel(0).reorder_levels(x.index.names)
        result = result.sort_index()
    else:
        result = pd.Series(dtype=float)
    
    return result

# 保持原有接口但使用向量化实现
def op_correlation(x, y, d):
    return op_correlation_vectorized(x, y, d)

def op_covariance(x, y, d):
    return op_covariance_vectorized(x, y, d)

def op_scale(x, a=1.0):
    # cross-sectional scale: returns a * x / sum(|x|)
    def _scale_group(g):
        denom = g.abs().sum()
        if denom == 0 or pd.isna(denom):
            return g * 0.0
        return a * g / denom
    return x.groupby(level=0, group_keys=False).apply(_scale_group)

def op_signedpower(x, a):
    return x.apply(lambda v: math.copysign((abs(v) ** a) if pd.notna(v) else np.nan, v))

def op_decay_linear(x, d):
    # weights descending from d..1 normalized
    weights = np.arange(d, 0, -1).astype(float)
    weights = weights / weights.sum()
    res_list = []
    for inst, gx in x.groupby(level="instrument"):
        vals = gx.values
        n = len(vals)
        out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            start = max(0, i - d + 1)
            seg = vals[start:i+1]
            w = weights[-len(seg):]  # align
            seg2 = np.where(np.isnan(seg), 0.0, seg)
            out[i] = float((seg2 * w).sum())
        res_list.append(pd.Series(out, index=gx.index))
    if len(res_list) == 0:
        return pd.Series(dtype=float)
    return pd.concat(res_list).sort_index()

def op_indneutralize(x, indclass):
    # Placeholder: industry mapping not provided.
    # To keep functionally correct, we return x unchanged but keep signature.
    # In real use, user must supply industry mapping and do cross-sectional regression.
    logger.debug("indneutralize called but no industry mapping provided; returning x unchanged.")
    return x

def preprocess_mad_zscore_vectorized(series: pd.Series):
    # 检查索引是否唯一，如果不唯一则重置索引
    if not series.index.is_unique:
        # 创建新的唯一索引
        # logger.info(f"{series.index} is not unique; dropping duplicates.")
        # 重置索引并去除重复项
        df_reset = pd.DataFrame({'value': series}).reset_index()
        df_reset = df_reset.drop_duplicates(subset=['datetime', 'instrument'], keep='first')
        # 重新设置索引
        df_reset = df_reset.set_index(['datetime', 'instrument'])
        series = df_reset['value']
    
    # 转换为DataFrame便于处理
    df = pd.DataFrame({'value': series}).reset_index()
    # 优化内存使用：将datetime列转换为category类型
    df['datetime'] = df['datetime'].astype('category')
    df['instrument'] = df['instrument'].astype('category')
    
    # 计算每个日期的中位数和MAD
    grouped = df.groupby('datetime')['value']
    medians = grouped.transform('median')
    mad = grouped.transform(lambda x: (x - x.median()).abs().median())
    
    # 处理MAD为0或NaN的情况
    mad = mad.where((mad > 0) & pd.notna(mad), 1.0)
    
    # 计算上下限并进行截断
    lower = medians - 5.0 * mad
    upper = medians + 5.0 * mad
    clipped = df['value'].clip(lower=lower, upper=upper)
    
    # 按日期进行Z-score标准化
    means = df.groupby('datetime')['value'].transform('mean')
    stds = df.groupby('datetime')['value'].transform('std')
    stds = stds.where((stds > 0) & pd.notna(stds), 1.0)
    
    # 构造结果Series
    result = (clipped - means) / stds
    result.index = series.index  # 恢复原始索引
    
    # 数据对齐：只保留非NaN的行
    result = result.dropna()
    
    return result

def preprocess_mad_zscore(series: pd.Series):
    return preprocess_mad_zscore_vectorized(series)

def rank_ic_mean_vectorized(series_factor: pd.Series, series_label: pd.Series):
    # 检查索引是否唯一，如果不唯一则重置索引
    if not series_factor.index.is_unique or not series_label.index.is_unique:
        # 创建新的唯一索引
        logger.info('series_factor or series_label index is not unique; dropping duplicates.')
        # 重置索引并去除重复项
        df_combined = pd.DataFrame({'f': series_factor, 'y': series_label}).reset_index()
        df_combined = df_combined.drop_duplicates(subset=['datetime', 'instrument'], keep='first')
        # 重新设置索引
        df_combined = df_combined.set_index(['datetime', 'instrument'])
        series_factor = df_combined['f']
        series_label = df_combined['y']
    
    # 转换为DataFrame
    try:
        df = pd.DataFrame({'f': series_factor, 'y': series_label}).reset_index()
        # 优化内存使用：将datetime列转换为category类型
        df['datetime'] = df['datetime'].astype('category')
        df['instrument'] = df['instrument'].astype('category')
    except Exception as e:
        logger.info(f"Error in rank_ic_mean_vectorized: {e}, series_factor: {series_factor}, series_label: {series_label}")
        raise e
    
    # 数据对齐：只保留两个序列都非NaN的行
    df = df.dropna(subset=['f', 'y'])
    
    # 对每个日期计算f和y的百分比排名
    df['f_rank'] = df.groupby('datetime')['f'].rank(pct=True)
    df['y_rank'] = df.groupby('datetime')['y'].rank(pct=True)
    
    # 过滤掉无变化的组
    valid_groups = df.groupby('datetime').filter(
        lambda g: g['f'].nunique() > 1 and g['y'].nunique() > 1
    )
    
    # 计算每个有效日期组的Spearman相关系数
    ic_results = []
    if not valid_groups.empty:
        # 按日期分组计算相关系数
        for dt, group in valid_groups.groupby('datetime'):
            if len(group) > 1:  # 确保有足够的数据点
                ic, _ = spearmanr(group['f_rank'], group['y_rank'])
                if not np.isnan(ic):
                    ic_results.append(ic)
    
    if len(ic_results) == 0:
        return float("nan"), np.array([])
    
    return float(np.nanmean(ic_results)), np.array(ic_results)

def rank_ic_mean(series_factor: pd.Series, series_label: pd.Series):
    return rank_ic_mean_vectorized(series_factor, series_label)