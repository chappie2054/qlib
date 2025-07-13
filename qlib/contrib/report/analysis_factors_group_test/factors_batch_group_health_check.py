import pandas as pd
import numpy as np
from typing import List, Tuple
from joblib import Parallel, delayed
from tqdm.notebook import tqdm


def _check_single_factor(
        col: Tuple[str, str],
        factor_series: pd.Series,
        datetimes: pd.Index,
        group_num: int,
        threshold: float
) -> dict:
    """
    快速检测某个因子在截面分组中的失败率（即 group 数小于 group_num 的频率）
    """
    df = factor_series.dropna().to_frame(name='factor')
    df = df.reset_index()

    # 排名（正向）
    df['rank'] = df.groupby('datetime')['factor'].rank(method='average', ascending=True)

    # 预分组结果容器
    group_counts = {}

    for dt, group in df.groupby('datetime'):
        if len(group) < group_num:
            group_counts[dt] = 0
            continue

        ranked = group['rank']
        # 使用 try/except 会慢，这里提前判断唯一值数量
        if ranked.nunique() < group_num:
            group_counts[dt] = ranked.nunique()
            continue

        try:
            bins = pd.qcut(ranked, q=group_num, labels=False, duplicates='drop')
            group_counts[dt] = bins.nunique()
        except Exception:
            group_counts[dt] = 0

    group_count_series = pd.Series(group_counts)
    fail_count = (group_count_series < group_num).sum()
    total_count = len(group_count_series)
    fail_ratio = fail_count / total_count if total_count > 0 else np.nan
    is_valid = fail_ratio <= threshold

    return {
        'factor': col[1],
        'total_dates': total_count,
        'fail_count': fail_count,
        'fail_ratio': fail_ratio,
        'is_valid': is_valid
    }


def check_factor_group_quality(
    factors_df: pd.DataFrame,
    group_num: int = 10,
    threshold: float = 0.05,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    并行检查因子的分组质量（能否成功分成 group_num 组）。

    参数:
        factors_df: MultiIndex ['datetime', 'instrument']，columns 为 MultiIndex [('feature', 因子名), ...]
        group_num: 分组数量
        threshold: 分组失败比例阈值
        n_jobs: 并行数，默认为全部核心

    返回:
        pd.DataFrame: 含每个因子的质量信息
    """
    assert isinstance(factors_df.columns, pd.MultiIndex)
    feature_cols = [col for col in factors_df.columns if col[0] == 'feature']
    datetimes = factors_df.index.get_level_values(0).unique()

    results = Parallel(n_jobs=n_jobs)(
        delayed(_check_single_factor)(
            col,
            factors_df[col].astype(np.float32),
            datetimes,
            group_num,
            threshold
        )
        for col in tqdm(feature_cols, desc='Checking Factor Quality')
    )

    # results = []
    #
    # for col in tqdm(feature_cols, desc='Checking Factor Quality'):
    #     result = _check_single_factor(
    #         col,
    #         factors_df[col].astype(np.float32),
    #         datetimes,
    #         group_num,
    #         threshold
    #     )
    #     results.append(result)

    result_df = pd.DataFrame(results).set_index('factor')
    return result_df
