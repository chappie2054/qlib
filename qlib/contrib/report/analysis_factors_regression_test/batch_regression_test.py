import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Dict, Any
from joblib import Parallel, delayed
from ..graph import ScatterGraph
from ..utils import guess_plotly_rangebreaks, _rankic_direction


def batch_factors_regression_test(
    factors_df: pd.DataFrame,
    risk_factors_names: Optional[List[str]] = None,
    n_jobs: int = -1, verbose: int = 10,
    show_notebook: bool = True
) -> pd.DataFrame:
    """对一批因子进行单因子回归测试，计算因子收益率和T值统计，并绘制收益率曲线

    采用向量化矩阵运算实现线性回归，替代statsmodels库以提升计算效率，并支持多因子并行计算

    参数:
        factors_df: 包含因子和标签数据的MultiIndex DataFrame
                    索引: ['datetime', 'instrument']
                    列: MultiIndex [('feature', 因子名), ('label', 标签名)]
        risk_factors_names: 风险因子名称列表，用于剥离风险因子暴露
        n_jobs: 并行计算的进程数，-1表示使用所有可用CPU
        verbose: 并行计算详细程度，数值越大输出越详细

    返回:
        pd.DataFrame: 包含各因子T值统计和分类结果的数据框，列包括:
                     - factor: 因子名称
                     - mean_abs_t: t值绝对值的均值
                     - ratio_abs_t_gt2: t值绝对值大于2的比例
                     - f_alpha_t_stat: 因子收益率序列的t检验统计量
                     - f_alpha_p_value: 因子收益率序列的t检验p值
                     - factor_type: 因子类型（收益类因子/风险类因子/无效因子）

    异常:
        ValueError: 当factors_df包含NaN值时抛出
        ValueError: 当factors_df没有或有多个标签列时抛出
        ValueError: 当没有待测试因子时抛出
        TypeError: 当输入参数类型不符合要求时抛出
    """
    # 参数类型验证
    if not isinstance(factors_df, pd.DataFrame):
        raise TypeError(f"factors_df must be a pandas DataFrame, got {type(factors_df).__name__}")
    if risk_factors_names is not None and not isinstance(risk_factors_names, list):
        raise TypeError(f"risk_factors_names must be a list or None, got {type(risk_factors_names).__name__}")
    if not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs must be an integer, got {type(n_jobs).__name__}")
    if not isinstance(verbose, int):
        raise TypeError(f"verbose must be an integer, got {type(verbose).__name__}")

    # 前提判断：检查是否存在NaN值
    if factors_df.isnull().any().any():
        raise ValueError("factors_df contains NaN values. Please clean the data before regression.")
    
    # 提取标签列（假设只有一个标签列）
    label_cols = factors_df.columns[factors_df.columns.get_level_values(0) == 'label']
    if len(label_cols) != 1:
        raise ValueError("factors_df must contain exactly one label column")
    label_col = label_cols[0]
    
    # 提取所有因子列
    feature_cols = factors_df.columns[factors_df.columns.get_level_values(0) == 'feature']
    all_factor_names = [col[1] for col in feature_cols]
    
    # 确定风险因子和待测试因子
    risk_factors = risk_factors_names if risk_factors_names is not None else []
    test_factors = [f for f in all_factor_names if f not in risk_factors]
    
    if not test_factors:
        raise ValueError("No test factors available after excluding risk factors")
    
    # 准备所有因子数据和标签数据
    all_features = factors_df.xs('feature', level=0, axis=1)
    label_data = factors_df.xs('label', level=0, axis=1).iloc[:, 0]  # 获取唯一的标签列
    
    # 存储每期回归结果
    regression_results = []
    
    # 获取所有唯一的时间戳
    datetimes = factors_df.index.get_level_values('datetime').unique()
    
    # 定义并行处理单个因子的函数
    def process_factor(factor: str) -> List[Dict[str, Any]]:
        factor_results = []
        # Calculate factor direction
        factor_series = all_features[factor]
        direction = _rankic_direction(factor_series, label_data)
        for datetime in datetimes:
            try:
                # 获取当前截面数据
                current_features = all_features.loc[datetime]
                current_label = label_data.loc[datetime]
                
                # 处理可能的NaN值
                valid_mask = ~current_features.isnull().any(axis=1) & ~current_label.isnull()
                current_features = current_features[valid_mask]
                current_label = current_label[valid_mask]
                
                if len(current_features) < 10:  # 确保有足够数据进行回归
                    continue
                
                # 构造自变量矩阵：当前因子 + 风险因子
                X = current_features[[factor]] * direction
                if risk_factors:
                    X = X.join(current_features[risk_factors])
                
                # 转换为numpy数组并添加常数项
                X_np = X.values
                X_np = np.hstack([np.ones((X_np.shape[0], 1)), X_np])  # 添加常数项
                y_np = current_label.values.reshape(-1, 1)
                
                # 使用矩阵运算求解线性回归
                X_T = X_np.T
                X_T_X = X_T @ X_np
                X_T_y = X_T @ y_np
                
                # 计算系数（处理奇异矩阵情况）
                try:
                    beta = np.linalg.inv(X_T_X) @ X_T_y
                except np.linalg.LinAlgError:
                    beta = np.linalg.pinv(X_T_X) @ X_T_y
                
                # 计算残差和MSE
                y_pred = X_np @ beta
                residuals = y_np - y_pred
                n, p = X_np.shape
                
                # 计算t值
                if n > p:
                    mse = np.sum(residuals ** 2) / (n - p)
                    cov_matrix = mse * np.linalg.inv(X_T_X) if np.linalg.det(X_T_X) !=0 else mse * np.linalg.pinv(X_T_X)
                    se_beta = np.sqrt(np.diag(cov_matrix))
                    t_values = beta.flatten() / se_beta
                else:
                    t_values = [np.nan] * p
                
                # 提取因子对应的系数和t值（常数项之后的第一个系数）
                factor_beta = beta[1, 0] if p > 1 else np.nan
                factor_t = t_values[1] if p > 1 and not np.isnan(t_values[1]) else np.nan
                
                factor_results.append({
                    'datetime': datetime,
                    'factor': factor,
                    'f_alpha': factor_beta,
                    't_value': factor_t
                })
            except Exception as e:
                print(f"Skipping factor {factor} at datetime {datetime} due to error: {e}")
                continue
        
        return factor_results
    
    # 使用joblib并行处理所有因子
    # 使用joblib并行处理所有因子
    regression_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_factor)(factor) for factor in test_factors
    )
    
    # 展平结果列表
    regression_results = [item for sublist in regression_results for item in sublist]
    
    # 转换为DataFrame并设置索引
    results_df = pd.DataFrame(regression_results)
    results_df.set_index('datetime', inplace=True)
    
    # 计算T值统计指标
    t_stats_list = []
    for factor in test_factors:
        factor_results = results_df[results_df['factor'] == factor]
        t_values = factor_results['t_value']
        f_alphas = factor_results['f_alpha']
        
        # 1. t值绝对值的均值
        mean_abs_t = t_values.abs().mean()
        
        # 2. t值绝对值大于2的比例
        ratio_gt2 = (t_values.abs() > 2).mean()
        
        # 3. 因子收益率序列的t检验
        t_test = stats.ttest_1samp(f_alphas, 0, nan_policy='omit')
        
        # 判断因子类型
        # if mean_abs_t > 1 and ratio_gt2 > 0.3:  # 经验阈值，可调整
        if mean_abs_t > 1 and ratio_gt2 > 0.0:  # 经验阈值，可调整
            if t_test.pvalue < 0.05:
                factor_type = '收益类因子'
            else:
                factor_type = '风险类因子'
        else:
            factor_type = '无效因子'
        
        t_stats_list.append({
            'factor': factor,
            'mean_abs_t': mean_abs_t,
            'ratio_abs_t_gt2': ratio_gt2,
            'f_alpha_t_stat': t_test.statistic,
            'f_alpha_p_value': t_test.pvalue,
            'factor_type': factor_type
        })
    
    t_stats_df = pd.DataFrame(t_stats_list)
    
    # 解析调仓周期（从label列名提取，假设格式为'label_72'）
    label_col = [col for col in factors_df.columns if col[0] == 'label'][0]
    try:
        rebalance_period = int(label_col[1].split('_')[1])
    except (IndexError, ValueError):
        raise ValueError(f"label 名 {label_col[1]} 不是合法格式，无法解析周期数")

    # 获取采样时间点（每rebalance_period取一个）
    unique_times = sorted(results_df.index.unique())
    selected_times = unique_times[::rebalance_period]
    sampled_results_df = results_df.loc[selected_times].copy()

    # 计算累积因子收益率
    sampled_results_df['cum_f_alpha'] = sampled_results_df.groupby('factor')['f_alpha'].cumsum()

    # 绘制累积因子收益率曲线
    all_figs = []
    for factor in test_factors:
        factor_data = sampled_results_df[sampled_results_df['factor'] == factor].reset_index()
        fig = ScatterGraph(
            df=factor_data,
            name_dict={'cum_f_alpha': 'Cumulative Factor Return'},
            layout=dict(
                title=f'{factor} Cumulative Factor Returns Over Time',
                xaxis=dict(
                    title='Date',
                    tickangle=45,
                    rangebreaks=guess_plotly_rangebreaks(factor_data['datetime'])
                ),
                yaxis=dict(title='Cumulative Factor Return')
            ),
            graph_kwargs={'mode': 'lines'}
        ).figure
        all_figs.append(fig)

    if show_notebook:
        ScatterGraph.show_graph_in_notebook(all_figs)

    # 整理结果DataFrame并设置因子为索引
    t_stats_df = t_stats_df.set_index('factor')
    return t_stats_df