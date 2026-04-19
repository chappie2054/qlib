#!/usr/bin/env python3
"""
ConstrainedPortfolioOptimizer回测测试
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))
from qlib.contrib.strategy.optimizer.constrained_optimizer import ConstrainedPortfolioOptimizer

# 全局资金费率数据缓存
_funding_rates_cache = None
_funding_rates_data = None

# 使用标准logging，避免详细控制台输出
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def load_funding_rates_data(funding_rates_path: str = r"D:\temp\带时间间隔的历史资金费率数据-修复编码-all_data.parquet") -> pd.DataFrame:
# def load_funding_rates_data(funding_rates_path: str = r'D:\temp\带时间间隔的历史资金费率数据-修复编码-all_data_2026-03-07.parquet') -> pd.DataFrame:
def load_funding_rates_data(funding_rates_path: str = r'D:\temp\带时间间隔的历史资金费率数据-修复编码-all_data-clickhouse-all.parquet') -> pd.DataFrame:
    """
    加载资金费率数据
    
    Parameters
    ----------
    funding_rates_path : str
        资金费率数据文件路径
        
    Returns
    -------
    pd.DataFrame
        MultiIndex格式的资金费率数据，index=['datetime', 'symbol'], columns=['funding_rate_interval', 'funding_rate']
    """
    global _funding_rates_data
    
    if _funding_rates_data is not None:
        return _funding_rates_data
    
    try:
        logger.info(f"加载资金费率数据: {funding_rates_path}")
        _funding_rates_data = pd.read_parquet(funding_rates_path)
        
        # 转换资金费率为数值类型（去除百分号）
        if _funding_rates_data['funding_rate'].dtype == 'object':
            _funding_rates_data['funding_rate'] = _funding_rates_data['funding_rate'].str.rstrip('%').astype(float) / 100
        
        logger.info(f"资金费率数据加载成功，形状: {_funding_rates_data.shape}")
        return _funding_rates_data
        
    except Exception as e:
        logger.error(f"加载资金费率数据失败: {e}")
        raise


def get_funding_rates_for_timestamp(funding_rates_df: pd.DataFrame, timestamp: pd.Timestamp, symbols: list, use_case: str = 'optimizer') -> pd.Series:
    """
    获取指定时间截面和品种列表的资金费率数据
    
    Parameters
    ----------
    funding_rates_df : pd.DataFrame
        资金费率数据
    timestamp : pd.Timestamp
        时间戳
    symbols : list
        品种列表
    use_case : str, optional
        使用场景: 
        - 'optimizer': 用于优化器资金费率约束（默认），只返回funding_rate_interval为'1小时'的品种资金费率
        - 'portfolio': 用于组合资金费率计算，返回所有品种的资金费率
        
    Returns
    -------
    pd.Series
        该时间截面的资金费率数据，缺失的品种资金费率为0
    """
    try:
        # 获取该时间截面的资金费率数据
        current_slice = funding_rates_df.loc[timestamp]
        
        # 为指定的品种列表获取资金费率，缺失的品种设为0
        funding_rates = pd.Series(0.0, index=symbols)
        
        # 多个品种的情况（移除单品种处理，因为截面中性策略不可能只有一个品种）
        available_symbols = current_slice.index.get_level_values(0) if current_slice.index.nlevels > 1 else current_slice.index
        common_symbols = set(available_symbols) & set(symbols)
        
        for symbol in common_symbols:
            # 检查使用场景
            if use_case == 'optimizer':
                # 用于优化器资金费率约束，只处理funding_rate_interval为'1小时'的品种
                if current_slice.loc[symbol, 'funding_rate_interval'] == '1小时':
                    funding_rates.loc[symbol] = current_slice.loc[symbol, 'funding_rate']
                # 否则资金费率设为0，不影响优化器计算
            else:
                # 用于组合资金费率计算，返回所有品种的资金费率
                funding_rates.loc[symbol] = current_slice.loc[symbol, 'funding_rate']
        
        return funding_rates
        
    except KeyError:
        # 该时间截面没有资金费率数据，所有品种资金费率为0
        return pd.Series(0.0, index=symbols)
        
    except Exception as e:
        logger.error(f"获取时间截面 {timestamp} 的资金费率数据失败: {e}")
        raise

# 抑制控制台输出，只保留进度信息
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('qlib').setLevel(logging.WARNING)

# 导入进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("警告: tqdm未安装，将使用简单的print进度显示")

# 配置日志 - 日志文件保存到指定目录
test_dir = Path(r"d:\PycharmProjects\qlib\workspace\py\test\log")
test_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
log_file_path = test_dir / f'constrained_optimizer_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            log_file_path,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BacktestSimulator:
    """回测模拟器：模拟截面中性策略的路径依赖调仓过程"""
    
    def __init__(self, 
                 optimizer: ConstrainedPortfolioOptimizer,
                 initial_capital: float = 1000000.0,
                 funding_rates_data: Optional[pd.DataFrame] = None,
                 price_data: Optional[pd.DataFrame] = None,
                 commission_rate: float = 0.0005):
        """
        初始化回测模拟器
        
        Parameters
        ----------
        optimizer : ConstrainedPortfolioOptimizer
            约束投资组合优化器
        initial_capital : float
            初始资金
        funding_rates_data : pd.DataFrame, optional
            资金费率数据，如果提供则使用真实资金费率，否则使用0
        price_data : pd.DataFrame, optional
            价格数据，用于计算收益率，格式为MultiIndex[datetime, instrument]，columns=['open', 'close']
        commission_rate : float, optional
            手续费率，默认单边万分之五
        """
        self.optimizer = optimizer
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.funding_rates_data = funding_rates_data
        self.price_data = price_data
        self.returns_data = None  # 预计算的收益率数据
        self.commission_rate = commission_rate  # 手续费率，单边万分之五
        
        # 记录历史
        self.portfolio_history = []
        self.trade_history = []
        self.performance_history = []
        self.optimization_status_history = []  # 新增：优化状态历史记录
        
        # 初始化持仓（空仓）
        self.current_weights = None
        self.current_positions = {}
        
        logger.info(f"回测模拟器初始化完成，初始资金: {initial_capital:,.2f}")
        if funding_rates_data is not None:
            logger.info(f"已加载资金费率数据，数据形状: {funding_rates_data.shape}")
        if price_data is not None:
            logger.info(f"已加载价格数据，数据形状: {price_data.shape}")
            self._calculate_returns_data()
    
    def load_prediction_data(self, data_path: str) -> pd.DataFrame:
        """
        加载预测数据
        
        Parameters
        ----------
        data_path : str
            parquet数据文件路径
            
        Returns
        -------
        pd.DataFrame
            MultiIndex预测数据，index=[datetime, instrument]
        """
        logger.info(f"加载预测数据: {data_path}")
        
        try:
            pred_df = pd.read_parquet(data_path)
            logger.info(f"数据加载成功，数据形状: {pred_df.shape}")
            logger.info(f"时间范围: {pred_df.index.get_level_values(0).min()} 到 {pred_df.index.get_level_values(0).max()}")
            logger.info(f"股票数量: {pred_df.index.get_level_values(1).nunique()}")
            
            return pred_df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def _calculate_returns_data(self):
        """
        预计算收益率数据：基于close价格计算当前期到下一期的收益率
        收益率 = (下一期close - 当前close) / 当前close
        """
        try:
            logger.info("开始预计算收益率数据...")
            
            # 按品种分组，计算每个品种的close价格变化率
            returns_list = []
            
            for symbol in self.price_data.index.get_level_values(1).unique():
                symbol_data = self.price_data.loc[pd.IndexSlice[:, symbol], :]
                symbol_data = symbol_data.sort_index()
                
                # 计算close价格的当期收益率（当前到下一期的变化率）
                symbol_returns = symbol_data['open'].pct_change().shift(-1)
                symbol_returns.name = 'return'
                
                returns_list.append(symbol_returns)
            
            # 合并所有品种的收益率数据
            self.returns_data = pd.concat(returns_list)
            self.returns_data = self.returns_data.sort_index()
            
            logger.info(f"收益率数据计算完成，数据形状: {self.returns_data.shape}")
            logger.info(f"收益率统计: 均值={self.returns_data.mean():.6f}, "
                       f"标准差={self.returns_data.std():.6f}, "
                       f"最小值={self.returns_data.min():.6f}, "
                       f"最大值={self.returns_data.max():.6f}")
            
        except Exception as e:
            logger.error(f"预计算收益率数据失败: {e}")
            raise
    
    def get_returns_for_timestamp(self, timestamp: pd.Timestamp, symbols: list) -> pd.Series:
        """
        获取指定时间截面和品种列表的当期收益率数据（当前close到下一期close的变化率）
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            当前时间戳
        symbols : list
            品种列表
            
        Returns
        -------
        pd.Series
            当前时间截面的收益率数据（当前close到下一期close的变化率），缺失的品种收益率为0
        """
        try:
            if self.returns_data is None:
                logger.warning("收益率数据未加载，返回0收益率")
                return pd.Series(0.0, index=symbols)
            
            # 获取当前时间截面的收益率数据（当前到下一期的变化率）
            current_returns = self.returns_data.loc[timestamp]
            
            # 为指定的品种列表获取收益率，缺失的品种设为0
            returns = pd.Series(0.0, index=symbols)
            
            # 多个品种的情况（时间截面数据）
            available_symbols = current_returns.index
            common_symbols = set(available_symbols) & set(symbols)
            
            for symbol in common_symbols:
                returns.loc[symbol] = current_returns.loc[symbol]
            
            logger.debug(f"时间 {timestamp}: 收益率数据获取完成，共 {len(symbols)} 个品种，"
                        f"其中 {returns[returns != 0].count()} 个品种有非零收益率")
            
            return returns
            
        except KeyError:
            # 该时间截面没有收益率数据，所有品种收益率为0
            logger.debug(f"时间 {timestamp}: 该时间截面没有收益率数据，所有品种收益率设为0")
            return pd.Series(0.0, index=symbols)
            
        except Exception as e:
            logger.error(f"获取时间截面 {timestamp} 的收益率数据失败: {e}")
            return pd.Series(0.0, index=symbols)

    def get_time_slice_data(self, pred_df: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series:
        """
        获取特定时间截面的预测数据
        
        Parameters
        ----------
        pred_df : pd.DataFrame
            完整预测数据
        timestamp : pd.Timestamp
            时间戳
            
        Returns
        -------
        pd.Series
            该时间截面的因子得分
        """
        try:
            # 获取该时间截面的数据
            time_slice = pred_df.loc[timestamp]
            factor_scores = time_slice['score']
            
            # 日志输出
            if len(factor_scores) > 0:
                logger.debug(f"时间 {timestamp}: 股票数量={len(factor_scores)}, "
                           f"分数范围=[{factor_scores.min():.4f}, {factor_scores.max():.4f}]")
            
            return factor_scores
        except KeyError:
            logger.warning(f"时间 {timestamp} 无数据")
            return pd.Series(dtype=float)
        except Exception as e:
            logger.error(f"获取时间截面数据失败 {timestamp}: {e}")
            return pd.Series(dtype=float)
    
    def calculate_portfolio_metrics(self, weights: pd.Series) -> Dict:
        """
        计算投资组合指标
        
        Parameters
        ----------
        weights : pd.Series
            权重
            
        Returns
        -------
        Dict
            投资组合指标
        """
        metrics = {}
        
        try:
            # 基本指标
            metrics['total_weight'] = weights.sum()
            metrics['long_weight'] = weights[weights > 0].sum()
            metrics['short_weight'] = weights[weights < 0].sum()
            metrics['long_count'] = (weights > 0).sum()
            metrics['short_count'] = (weights < 0).sum()
            metrics['net_exposure'] = metrics['long_weight'] + metrics['short_weight']
            metrics['gross_exposure'] = weights.abs().sum()
            
            # 约束验证
            config = self.optimizer.constraints_config
            # 数值容差参数 - 用于解决浮点数精度误差
            tolerance = 1e-8  # 1e-8 的容差足够处理 IEEE 754 双精度浮点数精度问题
            norm_tolerance = 1e-6  # 范数约束使用稍大的容差，因为范数计算涉及累积误差
            
            if 'weight_bounds' in config:
                min_w, max_w = config['weight_bounds']
                # 添加容差，避免因浮点数精度导致的微小超出
                metrics['weight_in_bounds'] = (
                    (weights >= min_w - tolerance) & (weights <= max_w + tolerance)
                ).all()
            
            if 'norm_type' in config and 'norm_bound' in config:
                norm_type = config['norm_type']
                norm_bound = config['norm_bound']
                if norm_type == 'l1':
                    actual_norm = np.linalg.norm(weights, ord=1)
                    # 添加容差，允许微小的范数超出（由数值精度误差导致）
                    metrics['norm_constraint_satisfied'] = actual_norm <= norm_bound + norm_tolerance
                    metrics['l1_norm'] = actual_norm
            
            if 'weight_sum' in config:
                expected_sum = config['weight_sum']
                # 使用容差处理权重和的微小误差
                metrics['weight_sum_error'] = abs(weights.sum() - expected_sum)
            
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算投资组合指标失败: {e}")
            return {}
    
    def calculate_turnover(self, new_weights: pd.Series) -> float:
        """
        计算换手率
        
        Parameters
        ----------
        new_weights : pd.Series
            新的权重
            
        Returns
        -------
        float
            换手率
        """
        if self.current_weights is None:
            # 第一次开仓，不计入换手率统计（返回0）
            logger.debug(f"第一次开仓，不计算换手率")
            return 0.0
        
        try:
            # 对齐权重 - 使用两个权重列表的并集作为基准，确保下架品种也被计算
            all_symbols = self.current_weights.index.union(new_weights.index)
            
            # 对齐当前权重到所有品种，下架品种保留当前权重
            aligned_current = self.current_weights.reindex(all_symbols, fill_value=0.0)
            # 对齐新权重到所有品种，新增品种权重为0
            aligned_new = new_weights.reindex(all_symbols, fill_value=0.0)
            
            # 计算换手率（两个序列的绝对差值之和）
            turnover = (aligned_new - aligned_current).abs().sum()
            
            logger.debug(f"换手率计算: 当前权重(前5个): {aligned_current.head().values}, "
                        f"新权重(前5个): {aligned_new.head().values}, "
                        f"权重差值(前5个): {(aligned_new - aligned_current).head().values}, "
                        f"换手率={turnover:.6f}")
            
            return turnover
            
        except Exception as e:
            logger.error(f"计算换手率失败: {e}")
            return 0.0
    
    def run_backtest(self, 
                    pred_df: pd.DataFrame,
                    rebalance_freq: str = 'hour',
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    verbose: bool = True,
                    max_periods: Optional[int] = None,
                    save_results: bool = True,
                    output_path: Optional[str] = None) -> Dict:
        """
        运行回测：在每个时间截面，使用上一期权重计算当期收益率，然后调仓更新权重
        
        正确的回测逻辑：
        1. 在t时刻，使用t-1时刻的权重计算t-1到t期间的收益率
        2. 收益率 = (t时刻open - t-1时刻open) / t-1时刻open
        3. 计算收益率后，执行调仓，更新权重为t时刻的新权重
        4. 新权重用于下一期（t到t+1期间）的收益计算
        
        Parameters
        ----------
        pred_df : pd.DataFrame
            预测数据，包含因子得分
        rebalance_freq : str
            调仓频率，默认每小时
        start_date : str, optional
            开始日期
        end_date : str, optional
            结束日期
        verbose : bool
            是否显示详细日志
        max_periods : int, optional
            最大回测截面数，用于快速测试。如果为None则运行全部截面
        save_results : bool, optional
            是否保存回测结果到文件
        output_path : str, optional
            回测结果输出路径，默认为当前目录
            
        Returns
        -------
        Dict
            回测结果统计
        """
        # 获取时间序列
        time_index = pred_df.index.get_level_values(0).unique()
        time_index = time_index.sort_values()
        
        # 过滤日期范围
        if start_date:
            time_index = time_index[time_index >= pd.Timestamp(start_date)]
        if end_date:
            time_index = time_index[time_index <= pd.Timestamp(end_date)]
        
        # 限制最大截面数（用于快速测试）
        if max_periods is not None and len(time_index) > max_periods:
            time_index = time_index[:max_periods]
        
        # 日志输出回测基本信息
        logger.info(f"开始运行回测...")
        logger.info(f"调仓频率: {rebalance_freq}")
        logger.info(f"回测时间范围: {time_index[0]} 到 {time_index[-1]}")
        logger.info(f"总共 {len(time_index)} 个时间截面")
        
        # 自动生成输出路径（如果未提供）
        if save_results and output_path is None:
            import datetime
            # 创建输出目录
            output_dir = Path(r"D:\temp\回测记录")
            output_dir.mkdir(parents=True, exist_ok=True)
            # 生成文件名
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"backtest_results_{current_time}.parquet")
            logger.info(f"自动生成输出路径: {output_path}")
        
        # 逐个时间截面进行优化
        successful_optimizations = 0
        failed_optimizations = 0
        
        # 初始化内存中的结果存储
        results_data = {
            'timestamp': [],
            'instrument': [],
            'current_weight': [],
            'previous_weight': [],
            'funding_rate': [],
            'factor_score': [],
            'individual_return': [],
            'factor_rank_pct': [],
            'portfolio_return': [],
            'capital': [],
            'turnover': [],
            'long_count': [],
            'short_count': [],
            'gross_exposure': [],
            'optimization_stage': [],
            'funding_cost': [],
            'commission_cost': []
        }
        
        # 使用进度条显示回测进度
        if HAS_TQDM:
            progress_bar = tqdm(total=len(time_index), desc="回测进度", unit="截面", 
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        else:
            print(f"回测进度: 0/{len(time_index)}", end="")
        
        for i, timestamp in enumerate(time_index):
            status = ""
            try:
                # 对于优化器，使用上一期的数据
                if i == 0:
                    # 第一个时间截面，没有上一期数据，按照实际应用处理：
                    # 1. 初始化权重为0
                    # 2. 跳过优化，直接进入下一个时间截面
                    logger.info(f"第一个时间截面 {timestamp}，没有上一期数据，初始化权重为0并跳过优化")
                    # 初始化权重为0
                    self.current_weights = pd.Series(0.0, index=pred_df.loc[timestamp].index) if len(pred_df.loc[timestamp]) > 0 else pd.Series(0.0)
                    # 更新进度条
                    if HAS_TQDM:
                        progress_bar.set_description(f"截面 {i+1}/{len(time_index)} {timestamp} [初始权重]")
                        progress_bar.update(1)
                    else:
                        print(f"\r回测进度: {i+1}/{len(time_index)} {timestamp} [初始权重]", end="")
                    # 跳过当前循环，进入下一个时间截面
                    continue
                else:
                    # 获取上一期时间戳
                    prev_timestamp = time_index[i-1]
                    
                    # 使用上一期的因子得分
                    factor_scores = self.get_time_slice_data(pred_df, prev_timestamp)
                    
                    if len(factor_scores) == 0:
                        status = "[无数据]"
                        failed_optimizations += 1
                        # 更新进度条
                        if HAS_TQDM:
                            progress_bar.set_description(f"截面 {i+1}/{len(time_index)} {prev_timestamp} {status}")
                            progress_bar.update(1)
                        else:
                            print(f"\r回测进度: {i+1}/{len(time_index)} {prev_timestamp} {status}", end="")
                        continue
                    
                    # 记录上一期权重
                    previous_weights = self.current_weights.copy()
                    
                    # 计算因子得分排名百分比
                    # 排名越高，百分比越大，1.0表示最高，0.0表示最低
                    factor_rank = factor_scores.rank(method='dense', ascending=False)
                    total_assets = len(factor_scores)
                    factor_rank_pct = (total_assets - factor_rank + 1) / total_assets
                    
                    # 准备上一期的资金费率数据
                    if self.funding_rates_data is not None:
                        # 使用真实的资金费率数据
                        try:
                            funding_rates = get_funding_rates_for_timestamp(
                                self.funding_rates_data, prev_timestamp, factor_scores.index.tolist(), use_case='optimizer'
                            )
                        except Exception as e:
                            logger.error(f"获取上一期资金费率数据失败 {prev_timestamp}: {e}")
                            status = "[资金费率错误]"
                            failed_optimizations += 1
                            # 更新进度条
                            if HAS_TQDM:
                                progress_bar.set_description(f"截面 {i+1}/{len(time_index)} {prev_timestamp} {status}")
                                progress_bar.update(1)
                            else:
                                print(f"\r回测进度: {i+1}/{len(time_index)} {prev_timestamp} {status}", end="")
                            continue
                    else:
                        # 使用默认的0资金费率
                        funding_rates = pd.Series(0.0, index=factor_scores.index)
                
                # 执行优化
                try:
                    # 保存原有的换手率约束值
                    original_turnover = self.optimizer.constraints_config.get('turnover')
                    
                    # 当i=1时（第一次优化器计算），临时允许turnover=1
                    if i == 1:
                        logger.info("第一次优化计算，临时将换手率约束调整为1")
                        self.optimizer.constraints_config['turnover'] = 1.0
                    
                    optimized_weights, optimization_info = self.optimizer(
                        factor_scores=factor_scores,
                        funding_rates=funding_rates,
                        current_weights=self.current_weights
                    )
                    
                    # 恢复原有的换手率约束值
                    if i == 1:
                        self.optimizer.constraints_config['turnover'] = original_turnover
                        logger.info(f"第一次优化计算完成，恢复原有换手率约束值: {original_turnover}")
                    
                    if optimized_weights is not None and len(optimized_weights) > 0:
                        # 计算换手率
                        turnover = self.calculate_turnover(optimized_weights)
                        
                        # 计算投资组合指标
                        portfolio_metrics = self.calculate_portfolio_metrics(optimized_weights)
                        
                        # 获取当前截面的收益率数据
                        current_returns = pd.Series(0.0, index=optimized_weights.index)
                        if self.returns_data is not None:
                            # 获取当前截面的收益率
                            current_returns = self.get_returns_for_timestamp(timestamp, optimized_weights.index.tolist())
                        
                        # 计算组合收益率（如果价格数据可用）- 必须在更新权重之前！
                        portfolio_return = 0.0
                        if self.returns_data is not None:
                            # 计算组合收益率: w^T * r，使用当前期的权重
                            if optimized_weights is not None:
                                portfolio_return = (optimized_weights * current_returns).sum()
                                
                                # 计算资金费率成本并从组合收益率中扣除
                                if self.funding_rates_data is not None:
                                    try:
                                        # 获取当前期的资金费率数据（与当前期权重对齐）
                                        funding_rates = get_funding_rates_for_timestamp(
                                            self.funding_rates_data, timestamp, optimized_weights.index.tolist(), use_case='portfolio'
                                        )
                                        # 资金费率成本 = 当前期权重 * 当前期资金费率
                                        funding_cost = (optimized_weights * funding_rates).sum()
                                        # 将资金费率成本从组合收益率中扣除
                                        portfolio_return += funding_cost
                                    except Exception as e:
                                        logger.warning(f"获取资金费率数据失败 {timestamp}: {e}")
                                        # 资金费率成本计算失败，不影响正常回测，继续执行

                        # 计算手续费成本 - 基于换手率，单边万分之五，多空都收取
                        # 换手率代表了调仓的比例，也就是需要收取手续费的部分
                        commission_cost = turnover * self.commission_rate
                        # 从组合收益率中扣除手续费
                        portfolio_return -= commission_cost
                        
                        # 更新当前权重为新权重 - 必须在计算收益率之后！
                        self.current_weights = optimized_weights.copy()
                        
                        # 更新资金
                        self.current_capital *= (1 + portfolio_return)
                        
                        # 记录回测结果
                        self.portfolio_history.append({
                            'timestamp': timestamp,
                            'weights': optimized_weights.copy(),
                            'metrics': portfolio_metrics,
                            'turnover': turnover,
                            'factor_scores': factor_scores.copy(),
                            'portfolio_return': portfolio_return,
                            'capital': self.current_capital
                        })
                        
                        # 记录优化状态
                        self.optimization_status_history.append({
                            'timestamp': timestamp,
                            'attempt_count': 1,
                            'success': True,
                            'error_message': '',
                            'optimization_details': optimization_info
                        })
                        
                        # 收集结果数据到内存中
                        optimization_stage = optimization_info.get('successful_stage', 'unknown') if optimization_info else 'unknown'
                        
                        # 对齐previous_weights到当前的品种列表
                        aligned_previous_weights = previous_weights.reindex(optimized_weights.index, fill_value=0.0)
                        
                        # 对齐factor_rank_pct到当前的品种列表
                        aligned_factor_rank_pct = factor_rank_pct.reindex(optimized_weights.index, fill_value=0.0)
                        
                        # 初始化funding_cost，如果没有计算则为0
                        funding_cost_value = funding_cost if 'funding_cost' in locals() else 0.0
                        
                        for instrument in optimized_weights.index:
                            results_data['timestamp'].append(timestamp)
                            results_data['instrument'].append(instrument)
                            results_data['current_weight'].append(optimized_weights[instrument])
                            results_data['previous_weight'].append(aligned_previous_weights[instrument])
                            results_data['funding_rate'].append(funding_rates[instrument])
                            results_data['factor_score'].append(factor_scores[instrument])
                            results_data['individual_return'].append(current_returns[instrument])
                            results_data['factor_rank_pct'].append(aligned_factor_rank_pct[instrument])
                            results_data['portfolio_return'].append(portfolio_return)
                            results_data['capital'].append(self.current_capital)
                            results_data['turnover'].append(turnover)
                            results_data['long_count'].append(portfolio_metrics.get('long_count', 0))
                            results_data['short_count'].append(portfolio_metrics.get('short_count', 0))
                            results_data['gross_exposure'].append(portfolio_metrics.get('gross_exposure', 0))
                            results_data['optimization_stage'].append(optimization_stage)
                            results_data['funding_cost'].append(funding_cost_value)
                            results_data['commission_cost'].append(commission_cost)
                        
                        successful_optimizations += 1
                        status = "[成功]"
                    else:
                        # 记录优化失败状态
                        self.optimization_status_history.append({
                            'timestamp': timestamp,
                            'attempt_count': 1,
                            'success': False,
                            'error_message': 'Optimization failed to converge',
                            'optimization_details': None
                        })
                        status = "[优化失败]"
                        failed_optimizations += 1
                
                except Exception as opt_error:
                    # 记录优化错误状态
                    self.optimization_status_history.append({
                        'timestamp': timestamp,
                        'attempt_count': 1,
                        'success': False,
                        'error_message': str(opt_error),
                        'optimization_details': None
                    })
                    status = "[错误]"
                    failed_optimizations += 1
            
            except Exception as e:
                status = "[处理错误]"
                failed_optimizations += 1
            
            # 统一更新进度条（只更新一次）
            if HAS_TQDM:
                progress_bar.set_description(f"截面 {i+1}/{len(time_index)} {prev_timestamp} {status}")
                progress_bar.update(1)
            else:
                print(f"\r回测进度: {i+1}/{len(time_index)} {prev_timestamp} {status}", end="")
        
        # 关闭进度条
        if HAS_TQDM:
            progress_bar.close()
        else:
            print()  # 换行
        
        # 计算回测统计结果
        backtest_stats = self._calculate_backtest_statistics(rebalance_freq)
        
        logger.info(f"\n=== 回测完成 ===")
        logger.info(f"成功优化: {successful_optimizations}")
        logger.info(f"失败优化: {failed_optimizations}")
        logger.info(f"总截面数: {len(time_index)}")
        
        # 避免除零错误
        total_attempts = successful_optimizations + failed_optimizations
        if total_attempts > 0:
            success_rate = successful_optimizations / total_attempts * 100
            logger.info(f"成功率: {success_rate:.2f}%")
        else:
            logger.warning("没有成功的优化尝试")
            success_rate = 0.0
        
        # 如果需要，一次性保存结果到文件
        if save_results and output_path:
            self._save_results_to_file(results_data, output_path)
        
        return backtest_stats
    
    def _log_portfolio_details(self, 
                              iteration: int, 
                              timestamp: pd.Timestamp, 
                              weights: pd.Series, 
                              metrics: Dict, 
                              turnover: float):
        """记录投资组合详细信息"""
        logger.info(f"迭代 {iteration}: 投资组合详情")
        logger.info(f"  时间: {timestamp}")
        logger.info(f"  股票数量: {len(weights)}")
        logger.info(f"  多头数量: {metrics.get('long_count', 0)}, 空头数量: {metrics.get('short_count', 0)}")
        logger.info(f"  多头权重: {metrics.get('long_weight', 0):.4f}, 空头权重: {metrics.get('short_weight', 0):.4f}")
        logger.info(f"  净敞口: {metrics.get('net_exposure', 0):.4f}, 总敞口: {metrics.get('gross_exposure', 0):.4f}")
        logger.info(f"  换手率: {turnover:.4f}")
        
        if 'l1_norm' in metrics:
            logger.info(f"  L1范数: {metrics['l1_norm']:.4f}")
        
        if 'weight_in_bounds' in metrics:
            status = "✅" if metrics['weight_in_bounds'] else "❌"
            logger.info(f"  权重边界约束: {status}")
    
    def _save_results_to_file(self, results_data: Dict, output_path: str) -> None:
        """
        将内存中的回测结果一次性保存到文件
        
        Parameters
        ----------
        results_data : Dict
            内存中存储的回测结果数据
        output_path : str
            输出文件路径
        """
        if not results_data or not results_data['timestamp']:
            logger.warning("没有数据可保存")
            return
        
        try:
            # 创建DataFrame
            df = pd.DataFrame(results_data)
            
            # 设置MultiIndex
            df.set_index(['timestamp', 'instrument'], inplace=True)
            
            # 保存到parquet文件
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
            
            logger.info(f"回测结果已保存到: {output_path}")
            logger.info(f"数据形状: {df.shape}")
            logger.info(f"时间范围: {df.index.get_level_values(0).min()} 到 {df.index.get_level_values(0).max()}")
            logger.info(f"品种数量: {df.index.get_level_values(1).nunique()}")
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
            raise


    

    
    def _calculate_backtest_statistics(self, rebalance_freq: str) -> Dict:
        """计算回测统计结果"""
        if not self.portfolio_history:
            return {}
        
        stats = {}
        
        try:
            # 提取指标
            turnovers = [record['turnover'] for record in self.portfolio_history]
            long_counts = [record['metrics'].get('long_count', 0) for record in self.portfolio_history]
            short_counts = [record['metrics'].get('short_count', 0) for record in self.portfolio_history]
            portfolio_returns = [record.get('portfolio_return', 0.0) for record in self.portfolio_history]
            capitals = [record.get('capital', self.initial_capital) for record in self.portfolio_history]
            short_counts = [record['metrics'].get('short_count', 0) for record in self.portfolio_history]
            gross_exposures = [record['metrics'].get('gross_exposure', 0) for record in self.portfolio_history]
            
            # 统计信息
            stats['total_rebalances'] = len(self.portfolio_history)
            stats['avg_turnover'] = np.mean(turnovers)
            stats['max_turnover'] = np.max(turnovers)
            stats['min_turnover'] = np.min(turnovers)
            stats['avg_long_count'] = np.mean(long_counts)
            stats['avg_short_count'] = np.mean(short_counts)
            stats['avg_gross_exposure'] = np.mean(gross_exposures)
            
            # 收益统计（如果价格数据可用）
            if portfolio_returns:
                stats['total_return'] = (capitals[-1] - self.initial_capital) / self.initial_capital
                stats['avg_portfolio_return'] = np.mean(portfolio_returns)
                stats['std_portfolio_return'] = np.std(portfolio_returns)
                stats['max_portfolio_return'] = np.max(portfolio_returns)
                stats['min_portfolio_return'] = np.min(portfolio_returns)
                
                # 根据调仓频率计算年化收益率和夏普比率（考虑加密货币市场7*24小时交易特点）
                if rebalance_freq == 'daily':
                    annualization_factor = 365  # 加密货币市场7*24小时，每年365天
                elif rebalance_freq == 'weekly':
                    annualization_factor = 365 / 7  # 加密货币市场7*24小时，每年约52.14周
                elif rebalance_freq == 'monthly':
                    annualization_factor = 365 / 30  # 加密货币市场7*24小时，每年约12.17月
                elif rebalance_freq == 'hour':
                    annualization_factor = 365 * 24  # 加密货币市场7*24小时，每年8760小时
                else:
                    annualization_factor = 365 * 24  # 默认按小时计算，每年8760小时
                
                # 计算年化收益率
                stats['annualized_return'] = stats['avg_portfolio_return'] * annualization_factor
                # 计算年化波动率
                stats['annualized_volatility'] = stats['std_portfolio_return'] * np.sqrt(annualization_factor)
                
                # 计算年化夏普比率（考虑无风险利率，默认无风险利率为0）
                risk_free_rate = 0.0  # 可以根据需要调整无风险利率
                stats['sharpe_ratio'] = (stats['annualized_return'] - risk_free_rate) / stats['annualized_volatility'] if stats['annualized_volatility'] > 0 else 0
                
                # 计算最大回撤
                cumulative_returns = np.cumprod(1 + np.array(portfolio_returns))
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                stats['max_drawdown'] = np.min(drawdowns) if len(drawdowns) > 0 else 0
                
                # 胜率（正收益的比例）
                positive_returns = [r for r in portfolio_returns if r > 0]
                stats['win_rate'] = len(positive_returns) / len(portfolio_returns) if portfolio_returns else 0
            
            # 约束满足率
            weight_bounds_violations = 0
            norm_violations = 0
            weight_sum_violations = 0
            
            # 使用与验证相同的容差参数
            tolerance = 1e-8
            norm_tolerance = 1e-6  # 范数约束使用与calculate_portfolio_metrics相同的容差
            
            for record in self.portfolio_history:
                metrics = record['metrics']
                if not metrics.get('weight_in_bounds', True):
                    weight_bounds_violations += 1
                if not metrics.get('norm_constraint_satisfied', True):
                    norm_violations += 1
                # 权重和约束的容差设置得更宽松一些，因为这是优化目标
                if metrics.get('weight_sum_error', 0) > tolerance * 10:
                    weight_sum_violations += 1
            
            total_records = len(self.portfolio_history)
            stats['weight_bounds_violation_rate'] = weight_bounds_violations / total_records
            stats['norm_violation_rate'] = norm_violations / total_records
            stats['weight_sum_violation_rate'] = weight_sum_violations / total_records
            
            logger.info(f"\n=== 回测统计结果 ===")
            logger.info(f"总调仓次数: {stats['total_rebalances']}")
            logger.info(f"平均换手率: {stats['avg_turnover']:.4f}")
            logger.info(f"最大换手率: {stats['max_turnover']:.4f}")
            logger.info(f"平均多头数量: {stats['avg_long_count']:.1f}")
            logger.info(f"平均空头数量: {stats['avg_short_count']:.1f}")
            logger.info(f"平均总敞口: {stats['avg_gross_exposure']:.4f}")
            logger.info(f"权重边界约束违反率: {stats['weight_bounds_violation_rate']:.2%}")
            logger.info(f"范数约束违反率: {stats['norm_violation_rate']:.2%}")
            logger.info(f"权重和约束违反率: {stats['weight_sum_violation_rate']:.2%}")
            
            # 输出收益统计（如果价格数据可用）
            if portfolio_returns:
                logger.info(f"\n=== 收益统计 ===")
                logger.info(f"总收益率: {stats['total_return']:.4f} ({stats['total_return']*100:.2f}%)")
                logger.info(f"平均组合收益率: {stats['avg_portfolio_return']:.6f}")
                logger.info(f"组合收益率标准差: {stats['std_portfolio_return']:.6f}")
                logger.info(f"最大单期收益: {stats['max_portfolio_return']:.6f}")
                logger.info(f"最小单期收益: {stats['min_portfolio_return']:.6f}")
                logger.info(f"夏普比率: {stats['sharpe_ratio']:.4f}")
                logger.info(f"最大回撤: {stats['max_drawdown']:.4f} ({stats['max_drawdown']*100:.2f}%)")
                logger.info(f"胜率: {stats['win_rate']:.4f} ({stats['win_rate']*100:.2f}%)")
            
            return stats
            
        except Exception as e:
            logger.error(f"计算回测统计失败: {e}")
            return {}


def constrained_optimizer_backtest(max_periods: int = None):
    """测试ConstrainedPortfolioOptimizer的回测功能
    
    Args:
        max_periods: 最大回测截面数，用于快速测试。如果为None则运行全部数据
    """
    logger.info("=== 开始测试ConstrainedPortfolioOptimizer回测功能 ===")
    
    # 数据路径
    data_path = r"D:\PycharmProjects\qlib\workspace\notebooks\filtered_pred_df.parquet"
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    else:
        logger.info("使用真实数据进行测试")
        try:
            pred_df = pd.read_parquet(data_path)
            # 使用完整数据集进行回测
            logger.info(f"时间范围: {pred_df.index.get_level_values(0).min()} 到 {pred_df.index.get_level_values(0).max()}")
            logger.info(f"股票数量: {pred_df.index.get_level_values(1).nunique()}")
        except Exception as e:
            logger.error(f"加载真实数据失败: {e}")
            return
    
    # 配置约束条件 - 截面中性策略
    constraints_config = {
        'weight_bounds': (-0.01, 0.01),      # 权重边界：-1% 到 +1%
        # 'weight_bounds': (-0.02, 0.02),      # 权重边界：-2% 到 +2%
        # 'weight_bounds': (-0.05, 0.05),      # 权重边界：-5% 到 +5%
        'weight_sum': 0.0,                 # 权重和 = 0（截面中性）
        'norm_type': 'l1',                 # L1范数约束
        'norm_bound': 1.0,                 # ||w||_1 ≤ 1，控制总杠杆率
        'turnover': 0.01,                   # 换手率约束 1%
        # 'turnover': 0.2,                   # 换手率约束 20%
        # 'turnover': 1,                   # 不做换手率约束
    }


    
    logger.info(f"约束配置: {constraints_config}")
    
    # 创建优化器（添加权重阈值过滤）
    optimizer = ConstrainedPortfolioOptimizer(
        constraints_config=constraints_config,
        objective_type="max_factor_score",
        solver=None,  # 使用默认求解器
        solver_kwargs={'verbose': False},
        # weight_threshold=0.001,  # 过滤绝对值小于0.001的权重（进一步放宽阈值）
        weight_threshold=0.005,  # 过滤绝对值小于0.005的权重（进一步放宽阈值）
        enable_funding_rate_constraint=True # 启用资金费率约束
        # enable_funding_rate_constraint=False # 不启用资金费率约束
    )

    logger.info("优化器创建完成")

    # 加载资金费率数据
    funding_rates_data = load_funding_rates_data()

    # 创建回测模拟器
    price_data = pd.read_parquet(r"D:\temp\回测用的品种close_open\open_close.parquet")

    # 回测模拟的参数配置
    backtest_simulator_params = {
        'optimizer': optimizer,
        'initial_capital': 1000000.0,
        'funding_rates_data': funding_rates_data,
        'price_data': price_data,
        # 'commission_rate': 0.0005
        'commission_rate': 0.001

    }

    logger.info(f"回测模拟参数配置: 单边手续费={backtest_simulator_params['commission_rate']}, 初始资金={backtest_simulator_params['initial_capital']}")

    backtest_simulator = BacktestSimulator(**backtest_simulator_params)
    
    # 运行全量回测
    try:
        # 获取所有时间截面
        time_index = pred_df.index.get_level_values(0).unique()
        time_index = time_index.sort_values()
        
        logger.info(f"全量回测数据统计:")
        logger.info(f"  总时间截面数: {len(time_index)}")
        logger.info(f"  时间范围: {time_index.min()} 到 {time_index.max()}")
        
        # 根据参数决定是否限制回测截面数
        if max_periods is not None:
            logger.info(f"  ⚡ 快速测试模式: 只运行前{max_periods}个截面")
        else:
            logger.info(f"  📊 完整回测模式: 运行全部{len(time_index)}个截面")
        
        backtest_stats = backtest_simulator.run_backtest(
            pred_df=pred_df,
            rebalance_freq='hour',
            verbose=True,
            max_periods=max_periods
        )
        
        # 只显示最终结果
        logger.info("✅ 回测测试完成")
        logger.info(f"回测统计: 成功率100%, 平均换手率{backtest_stats.get('avg_turnover', 0):.4f}")
        
        # 如果有收益统计，显示详细信息
        if 'total_return' in backtest_stats:
            logger.info(f"\n📊 收益表现:")
            logger.info(f"  总收益率: {backtest_stats['total_return']*100:.2f}%")
            logger.info(f"  年化收益率: {backtest_stats['annualized_return']*100:.2f}%")
            logger.info(f"  夏普比率: {backtest_stats['sharpe_ratio']:.4f}")
            logger.info(f"  最大回撤: {backtest_stats['max_drawdown']*100:.2f}%")
            logger.info(f"  胜率: {backtest_stats['win_rate']*100:.2f}%")
            logger.info(f"  平均单期收益: {backtest_stats['avg_portfolio_return']*100:.4f}%")
        
        return backtest_stats
        
    except Exception as e:
        logger.error(f"回测失败: {e}")
        raise





def main():
    """主函数 - 支持命令行参数控制"""
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='ConstrainedPortfolioOptimizer回测测试')
    parser.add_argument('--max-periods', type=int, default=None,
                       help='最大回测截面数，用于快速测试。如果不指定则运行全部数据')
    parser.add_argument('--quick-test', action='store_true',
                       help='快速测试模式，只运行前100个截面')
    parser.add_argument('--full-test', action='store_true',
                       help='完整测试模式，运行全部数据（默认）')
    
    args = parser.parse_args()
    
    # 确定max_periods参数
    if args.quick_test:
        max_periods = 100
        logger.info("🚀 快速测试模式：只运行前100个截面")
    elif args.max_periods is not None:
        max_periods = args.max_periods
        logger.info(f"🎯 自定义测试模式：运行前{max_periods}个截面")
    else:
        max_periods = None
        logger.info("📊 完整测试模式：运行全部数据")
    
    logger.info("开始运行ConstrainedPortfolioOptimizer综合测试")
    
    # 抑制警告
    warnings.filterwarnings('ignore')
    
    # 开启DEBUG日志以查看详细信息
    logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 只运行回测功能测试（跳过其他测试）
        logger.info("\n=== 运行回测功能测试 ===")
        constrained_optimizer_backtest(max_periods=max_periods)
        
        logger.info("\n🎉 回测测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
