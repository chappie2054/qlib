# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .base import BaseOptimizer

class TurnoverConstrainedOptimizer(BaseOptimizer):
    """
    换手率约束优化器 - 基于品种数量比例的换手约束
    
    该优化器不采用传统的二次规划方法约束权重变化，
    而是通过控制换掉多少比例的品种来实现换手率约束。
    
    注意：该优化器支持空头股票数量与多头股票数量的比例由参数控制。
    """
    
    def __init__(self, max_turnover_rate: float = 0.1, short_long_ratio: float = 1.0):
        """
        初始化换手率约束优化器
        
        Parameters
        ----------
        max_turnover_rate : float, optional
            最大换手率（相对于n_group的比例），默认0.1即10%
        short_long_ratio : float, optional
            空头端品种数量与多头端品种数量的比例，默认为1.0（即空头数量是多头数量的两倍）
        """
        self.max_turnover_rate = max_turnover_rate
        self.short_long_ratio = short_long_ratio
    
    def _validate_and_return_positions(self, new_long_stocks: List[str], new_short_stocks: List[str], n_group: int) -> Tuple[List[str], List[str]]:
        """
        验证新的多空组合并返回结果
        
        Parameters
        ----------
        new_long_stocks : List[str]
            新的多头股票列表
        new_short_stocks : List[str]
            新的空头股票列表
        n_group : int
            预期的多头股票数量
            
        Returns
        -------
        Tuple[List[str], List[str]]
            验证通过的多空股票列表
            
        Raises
        ------
        ValueError
            当验证失败时抛出异常
        """
        # 检查数量是否符合要求
        if len(new_long_stocks) != n_group:
            raise ValueError(f"多头应该有{n_group}只股票，实际有{len(new_long_stocks)}只")
        
        # 检查是否有重叠品种
        overlap = set(new_long_stocks) & set(new_short_stocks)
        if len(overlap) > 0:
            raise ValueError(f"多头和空头存在重叠股票: {overlap}")
        
        return new_long_stocks, new_short_stocks

    def __call__(
        self,
        score: pd.Series,
        current_position: Dict[str, float],
        n_group: int,
        **kwargs
    ) -> Tuple[List[str], List[str]]:
        """
        根据换手率约束生成新的多空股票列表
        
        Parameters
        ----------
        score : pd.Series
            股票得分，index为股票代码
        current_position : Dict[str, float]
            当前持仓，key为股票代码，value为持仓金额（正为多头，负为空头）
        n_group : int
            当前的多空分组数量
            
        Returns
        -------
        Tuple[List[str], List[str]]
            新的多头股票列表和空头股票列表
        """
        # 1. 获取排序后的股票列表（提前处理，避免重复）
        score_sorted = score.sort_values(ascending=False)
        all_stocks = list(score_sorted.index)
        
        # 2. 获取理想的多空股票（统一处理）
        ideal_long = score_sorted.head(n_group).index.tolist()
        # 空头股票数量由short_long_ratio参数控制
        n_short_group = int(n_group * self.short_long_ratio)
        ideal_short = score_sorted.tail(n_short_group).index.tolist()
        
        # 3. 检查股票数量是否充足
        if len(all_stocks) < n_group + n_short_group:
            raise ValueError(f"股票数量不足：需要至少{n_group + n_short_group}只股票，实际只有{len(all_stocks)}只")
        
        # 4. 检查理想的多头和空头是否有重叠
        if set(ideal_long) & set(ideal_short):
            raise ValueError(f"股票数量不足：多头和空头存在重叠")
        
        # 5. 获取当前持仓（**保留所有股票，包括下架的**）
        last_long_stocks = [stock for stock, amount in current_position.items() if amount > 0]
        last_short_stocks = [stock for stock, amount in current_position.items() if amount < 0]
        
        # 6. 检查持仓平衡性
        if (not last_long_stocks and last_short_stocks) or (last_long_stocks and not last_short_stocks):
            long_count = len(last_long_stocks)
            short_count = len(last_short_stocks)
            raise ValueError(f"持仓状态不平衡：多头{long_count}只，空头{short_count}只")
        
        # # 检查当前持仓是否符合short_long_ratio比例
        # if last_long_stocks and last_short_stocks:
        #     expected_short_count = int(len(last_long_stocks) * self.short_long_ratio)
        #     if len(last_short_stocks) != expected_short_count:
        #         raise ValueError(f"当前持仓不符合short_long_ratio比例：多头{len(last_long_stocks)}只，"
        #                          f"空头{len(last_short_stocks)}只，预期空头数量应为{expected_short_count}只"
        #                          f"（比例{self.short_long_ratio}）")
        
        # 7. 如果没有持仓，直接返回理想股票列表
        if not last_long_stocks and not last_short_stocks:
            return self._validate_and_return_positions(ideal_long, ideal_short, n_group)
        
        # 8. 计算最大换手股票数量
        max_turnover_stocks_long = max(1, int(n_group * self.max_turnover_rate))
        max_turnover_stocks_short = max(1, int(n_short_group * self.max_turnover_rate))
        
        # 9. 构建新的多头列表（**修复逻辑错误**）
        # 获取当前持仓中有效的多头股票（未下架的）
        valid_last_long = [s for s in last_long_stocks if s in score.index]
        
        # 修复：检查理想多头是否是当前持仓的子集（包括下架的）
        if set(ideal_long).issubset(set(last_long_stocks)):
            # 情况1：理想多头是当前持仓的子集（包括下架的）
            # 只需要从当前持仓中选择排名最高的n_group只股票
            valid_current_scores = score[valid_last_long].sort_values(ascending=False)
            new_long_stocks = valid_current_scores.head(n_group).index.tolist()
        else:
            # 情况2：需要替换和补充（**自动处理下架品种**）
            # 优先保留未下架的股票
            valid_scores = score[valid_last_long].sort_values(ascending=False)
            long_to_keep = valid_scores.head(n_group - max_turnover_stocks_long).index.tolist()
            
            # 从理想多头中补充，排除已保留的
            ideal_long_scores = score[ideal_long].sort_values(ascending=False)
            available_new = [s for s in ideal_long_scores.index if s not in long_to_keep]
            new_long_stocks = long_to_keep + available_new[:max_turnover_stocks_long]
            
            new_long_stocks = new_long_stocks[:n_group]
        
        # 10. 构建新的空头列表（**修复逻辑错误**）
        # 获取当前持仓中有效的空头股票（未下架的）
        valid_last_short = [s for s in last_short_stocks if s in score.index]
        
        # 修复：检查理想空头是否是当前持仓的子集（包括下架的）
        if set(ideal_short).issubset(set(last_short_stocks)):
            # 情况1：理想空头是当前持仓的子集（包括下架的）
            # 只需要从当前持仓中选择排名最低的n_short_group只股票
            valid_current_scores = score[valid_last_short].sort_values(ascending=True)
            new_short_stocks = valid_current_scores.head(n_short_group).index.tolist()
        else:
            # 情况2：需要替换和补充（**自动处理下架品种**）
            # 优先保留未下架的股票
            valid_scores = score[valid_last_short].sort_values(ascending=True)
            short_to_keep = valid_scores.head(n_short_group - max_turnover_stocks_short).index.tolist()
            
            # 从理想空头中补充，排除已保留的
            ideal_short_scores = score[ideal_short].sort_values(ascending=True)
            available_new = [s for s in ideal_short_scores.index if s not in short_to_keep and s not in new_long_stocks]
            new_short_stocks = short_to_keep + available_new[:max_turnover_stocks_short]
            
            new_short_stocks = new_short_stocks[:n_short_group]
            
            # 最终检查：确保多头和空头没有重叠
            overlap = set(new_long_stocks) & set(new_short_stocks)
            if overlap:
                raise ValueError(f"多头和空头存在重叠股票: {overlap}")
        
        # 11. 使用封装的验证方法返回结果
        return self._validate_and_return_positions(new_long_stocks, new_short_stocks, n_group)