# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
通用约束投资组合优化器

该模块提供了ConstrainedPortfolioOptimizer类，支持灵活的约束配置和优化的投资组合权重计算。
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp

from .base import BaseOptimizer

logger = logging.getLogger(__name__)


class ConstrainedPortfolioOptimizer(BaseOptimizer):
    """
    通用约束投资组合优化器
    
    目标函数：max (w_t * F_t)
    
    约束条件：
    - weight_bounds: 权重边界约束 (w_s ≤ w_t ≤ w_l)
    - weight_sum: 权重和约束 (∑w_t = 0)
    - norm_type: 范数约束类型 ('l1')
    - norm_bound: 范数约束上界 (||w_t||_1 ≤ τ)，**必须指定，不能为None**
    - turnover: 换手率约束 (∑|w_t - w_{t-1}| ≤ TVR_thresh)
    
    自动约束：
    - 资金费率约束 (w_t^⊤ R_t ≥ 0): 确保组合资金费率非负
    
    权重阈值过滤（可选）：
    - weight_threshold: 权重阈值过滤，将小于阈值的权重设为0
    
    示例约束配置：
    {
        'weight_bounds': (-0.1, 0.1),           # 权重边界
        'weight_sum': 0.0,                      # 权重和 = 0
        'norm_type': 'l1',                      # L1范数约束
        'norm_bound': 1.0,                      # ||w||_1 ≤ 1，必须指定
        'turnover': 0.2,                        # 换手率约束
    }
    
    示例初始化（带权重阈值过滤）：
    optimizer = ConstrainedPortfolioOptimizer(
        constraints_config={...},
        weight_threshold=0.009  # 过滤绝对值小于0.009的权重
    )
    
    重要说明：
    1. 该优化器是**路径依赖**的，当前权重优化结果依赖于之前的权重状态
    2. **不能同时求解多个截面的权重优化问题**，每次只能处理一个截面
    3. 使用cvxpy进行凸优化求解，当cvxpy不可用时会抛出ImportError
    4. 范数约束 `norm_bound` 必须指定，且不能为None
    """
    
    def __init__(
        self,
        constraints_config: Dict,
        objective_type: str = "max_factor_score",
        solver: Optional[str] = None,
        solver_kwargs: Optional[Dict] = None,
        weight_threshold: Optional[float] = None,
        enable_funding_rate_constraint: bool = False,
        **kwargs
    ):
        """
        初始化通用约束投资组合优化器
        
        Parameters
        ----------
        constraints_config : Dict
            约束配置字典，详见类文档说明
        objective_type : str, optional
            目标函数类型，默认为"max_factor_score"（最大化因子得分）
            可选值：
            - "max_factor_score": 最大化因子得分
        solver : str, optional
            cvxpy求解器，默认自动选择
        solver_kwargs : dict, optional
            求解器参数
        weight_threshold : float, optional
            权重阈值，用于过滤小于该阈值的权重（绝对值）
            如果为None，则不进行阈值过滤
        enable_funding_rate_constraint : bool, optional
            是否开启资金费率约束，默认为False（关闭）
        """
        # 验证norm_bound不能为None
        if 'norm_bound' not in constraints_config or constraints_config['norm_bound'] is None:
            raise ValueError("约束配置中必须指定norm_bound，且不能为None")
        
        self.constraints_config = constraints_config
        self.objective_type = objective_type
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}
        self.weight_threshold = weight_threshold
        self.enable_funding_rate_constraint = enable_funding_rate_constraint
    

    
    def _build_objective(
        self,
        w: cp.Variable,
        factor_scores: np.ndarray
    ):
        """构建目标函数"""
        if self.objective_type == "max_factor_score":
            # 最大化因子得分
            return cp.Maximize(w @ factor_scores)
        else:
            raise ValueError(f"不支持的目标函数类型: {self.objective_type}")
    
    def _validate_solution(self, weights: pd.Series, stage_config: Dict = None):
        """验证优化解是否满足约束，不满足则报错"""
        # 使用当前阶段的约束配置，如果没有提供则使用原始配置
        config = stage_config if stage_config is not None else self.constraints_config
        tolerance = 2e-3
        
        # 验证权重边界
        min_weight, max_weight = config.get('weight_bounds', (-1.0, 1.0))
        if not (weights >= min_weight - tolerance).all() or not (weights <= max_weight + tolerance).all():
            raise ValueError(f"权重边界约束未满足: 最小值 {weights.min()}, 最大值 {weights.max()}, 预期范围 [{min_weight}, {max_weight}]")
        
        # 验证权重和约束
        total_weight = config.get('weight_sum', 0.0)
        actual_sum = weights.sum()
        if abs(actual_sum - total_weight) > tolerance:
            raise ValueError(f"权重和约束未满足: ∑w = {actual_sum}, 预期 = {total_weight}")
        
        # 验证范数约束
        norm_type = config.get('norm_type', 'l1')
        norm_bound = config.get('norm_bound', 1.0)
        if norm_type == 'l1':
            actual_norm = np.linalg.norm(weights, ord=1)
            if actual_norm > norm_bound + tolerance:
                raise ValueError(f"范数约束未满足: ||w||_{norm_type} = {actual_norm} > {norm_bound}")
        
        # 验证换手率约束
        if 'turnover' in config and hasattr(self, '_current_weights') and self._current_weights is not None:
            max_turnover = config.get('turnover', None)
            # 如果换手率约束为None，表示不做限制，跳过验证
            if max_turnover is not None:
                aligned_current = self._current_weights.reindex(weights.index, fill_value=0.0)
                actual_turnover = (weights - aligned_current).abs().sum()
                if actual_turnover > max_turnover + tolerance:
                    raise ValueError(f"换手率约束未满足: 实际换手率 {actual_turnover}, 预期最大值 {max_turnover}")
        
        # 验证资金费率约束
        if self.enable_funding_rate_constraint and hasattr(self, '_funding_rates'):
            funding_rates_aligned = self._funding_rates.reindex(weights.index, fill_value=0.0)
            actual_funding_rate = weights @ funding_rates_aligned
            if actual_funding_rate < -tolerance:
                raise ValueError(f"资金费率约束未满足: 实际资金费率 {actual_funding_rate}, 预期 ≥ 0")
    
    def _apply_weight_threshold_filter(self, weights: pd.Series) -> pd.Series:
        """
        应用权重阈值过滤，将小于阈值的权重设为0，并保持原始权重和
        
        Parameters
        ----------
        weights : pd.Series
            原始权重
            
        Returns
        -------
        pd.Series
            过滤后的权重（保持原始权重和）
        """
        if self.weight_threshold is None:
            return weights
        
        # 复制权重以避免修改原始数据
        filtered_weights = weights.copy()
        
        # 保存原始权重和
        original_sum = weights.sum()
        
        # 过滤小于阈值的权重（绝对值）
        mask = np.abs(filtered_weights) < self.weight_threshold
        filtered_weights[mask] = 0.0
        
        # # 调整剩余权重，保持原始权重和
        # non_zero_mask = ~mask
        # if non_zero_mask.sum() > 0:
        #     current_sum = filtered_weights.sum()
        #     if current_sum != 0:
        #         scaling_factor = original_sum / current_sum
        #         filtered_weights[non_zero_mask] *= scaling_factor
        
        return filtered_weights
    
    def __call__(self, 
                 factor_scores: pd.Series,
                 funding_rates: pd.Series,
                 current_weights: Optional[pd.Series] = None,
                 optimization_info: Optional[Dict] = None) -> Tuple[pd.Series, Dict]:
        """
        执行投资组合优化
        
        Parameters
        ----------
        factor_scores : pd.Series
            因子得分
        funding_rates : pd.Series
            资金费率
        current_weights : pd.Series, optional
            当前权重（用于换手率约束）
        optimization_info : Dict, optional
            优化信息字典，用于记录详细的优化过程
            
        Returns
        -------
        pd.Series
            优化后的权重
        Dict
            优化信息，包含成功状态和详细信息
        """
        if optimization_info is None:
            optimization_info = {}
        
        # 初始化优化信息
        optimization_info.update({
            'optimization_success': False,
            'optimization_type': 'unknown',
            'solver_status': 'unknown',
            'successful_stage': 'unknown',
            'total_attempts': 0,
            'failed_stages': [],
            'attempts': []  # 详细的尝试记录
        })
        
        # 对齐数据 - 只根据因子得分的缺失来判断品种是否下架
        # 1. 首先过滤出因子得分非缺失的品种（这些是有效的活跃品种）
        valid_factor_scores = factor_scores.dropna()
        
        # 2. 将资金费率与有效品种对齐
        funding_rates_aligned = funding_rates.reindex(valid_factor_scores.index)
        
        # 3. 填充资金费率缺失值为0（无论是因为对齐还是原本就缺失）
        funding_rates_aligned = funding_rates_aligned.fillna(0.0)
        
        # 4. 构建对齐后的数据，确保只有有效品种
        aligned_data = pd.DataFrame({
            'factor_scores': valid_factor_scores,
            'funding_rates': funding_rates_aligned
        })
        
        factor_scores_aligned = aligned_data['factor_scores']
        funding_rates_aligned = aligned_data['funding_rates']
        

        
        n_assets = len(factor_scores_aligned)
        if n_assets == 0:
            error_message = "没有有效的资产进行优化，这说明数据有问题"
            logger.error(error_message)
            raise RuntimeError(error_message)
        
        # 优化阶段配置：只放宽换手率约束，其他约束保持不变
        original_turnover = self.constraints_config.get('turnover', None)
        norm_bound = self.constraints_config.get('norm_bound')  # norm_bound是必须指定的
        
        stages = []
        
        if original_turnover is not None:
            # 逐步放宽换手率约束，从1倍开始，直到original_turnover * 倍率 > norm_bound
            max_stage = 100  # 防止无限循环的安全边界
            for multiplier in range(1, max_stage + 1):
                # 计算当前倍率下的换手率约束
                current_turnover = original_turnover * multiplier
                
                # 如果当前换手率超过norm_bound，则停止
                if current_turnover > norm_bound:
                    break
                
                # 添加当前阶段，保留所有约束，只修改换手率
                stages.append({
                    'name': f'换手率放宽{multiplier}倍',
                    'weight_bounds': self.constraints_config['weight_bounds'],
                    'weight_sum': self.constraints_config['weight_sum'],
                    'norm_bound': norm_bound,
                    'turnover_bound': current_turnover,
                    'description': f'换手率放宽{multiplier}倍'
                })
            
            # 注意：不添加完全放开约束的阶段，只到接近norm_bound的位置
        else:
            # 如果没有原始换手率约束，只使用一个阶段
            stages.append({
                'name': '无换手率约束',
                'weight_bounds': self.constraints_config['weight_bounds'],
                'weight_sum': self.constraints_config['weight_sum'],
                'norm_bound': norm_bound,
                'turnover_bound': None,
                'description': '无换手率约束'
            })
        
        total_attempts = 0
        failed_stages = []
        attempts = []
        
        # 保存当前权重和资金费率，用于验证
        self._current_weights = current_weights
        self._funding_rates = funding_rates_aligned
        
        # 尝试每个阶段
        for stage_idx, stage in enumerate(stages):
            stage_name = stage['name']
            
            try:
                total_attempts += 1
                attempt_info = {
                    'stage': stage_name,
                    'stage_idx': stage_idx,
                    'attempt_number': total_attempts,
                    'success': False,
                    'error': None,
                    'solver_status': 'unknown',
                    'objective_value': None
                }
                
                logger.info(f"尝试优化阶段 {stage_idx + 1}/{len(stages)}: {stage_name}")
                
                # 构建优化问题
                problem, w = self._build_optimization_problem(
                    factor_scores_aligned,
                    funding_rates_aligned,
                    current_weights,
                    stage
                )
                
                # 求解优化问题
                problem.solve(solver=self.solver, **self.solver_kwargs)
                
                attempt_info['solver_status'] = problem.status
                attempt_info['objective_value'] = problem.value
                
                if problem.status == cp.OPTIMAL:
                    # 优化成功
                    weights = pd.Series(w.value, index=factor_scores_aligned.index)
                    
                    # 应用权重阈值过滤（在验证之前）
                    weights = self._apply_weight_threshold_filter(weights)
                    
                    # 验证约束满足情况，传递当前阶段的配置
                    # 将stage配置转换为_validate_solution需要的格式
                    validate_config = {
                        'weight_bounds': stage['weight_bounds'],
                        'weight_sum': stage['weight_sum'],
                        'norm_type': self.constraints_config.get('norm_type', 'l1'),
                        'norm_bound': stage['norm_bound'],
                        'turnover': stage['turnover_bound']
                    }
                    self._validate_solution(weights, validate_config)
                    
                    # 完全满足约束
                    optimization_info['optimization_success'] = True
                    optimization_info['successful_stage'] = stage_name
                    optimization_info['total_attempts'] = total_attempts
                    optimization_info['failed_stages'] = failed_stages
                    optimization_info['attempts'] = attempts
                    optimization_info['solver_status'] = problem.status
                    
                    logger.info(f"优化成功完成于阶段: {stage_name}")
                    return weights, optimization_info
                else:
                    # 求解器失败
                    attempt_info['success'] = False
                    attempt_info['error'] = f"求解器状态: {problem.status}"
                    failed_stages.append(stage_name)
                    
                    logger.warning(f"阶段 {stage_name} 求解失败，状态: {problem.status}")
                
                attempts.append(attempt_info)
                
            except Exception as e:
                # 阶段执行失败
                attempt_info['success'] = False
                attempt_info['error'] = str(e)
                failed_stages.append(stage_name)
                attempts.append(attempt_info)
                
                logger.error(f"阶段 {stage_name} 执行失败: {e}")
        
        # 所有阶段都失败，直接报错
        error_message = f"所有{len(stages)}个优化阶段均失败，无法求出解。请检查约束条件或因子得分。"
        logger.error(error_message)
        optimization_info['optimization_success'] = False
        optimization_info['total_attempts'] = total_attempts
        optimization_info['failed_stages'] = failed_stages
        optimization_info['attempts'] = attempts
        optimization_info['solver_status'] = 'failed'
        
        raise RuntimeError(error_message)
    


    def _build_optimization_problem(self, factor_scores: pd.Series, funding_rates: pd.Series,
                                   current_weights: Optional[pd.Series], stage: Dict) -> Tuple[cp.Problem, cp.Variable]:
        """构建优化问题"""
        n_assets = len(factor_scores)
        w = cp.Variable(n_assets)
        
        # 目标函数：最大化因子得分
        objective = cp.Maximize(w @ factor_scores.values)
        
        # 约束条件
        constraints = []
        
        # 1. 权重边界约束
        if stage['weight_bounds'] is not None:
            min_w, max_w = stage['weight_bounds']
            constraints.extend([w >= min_w, w <= max_w])
        
        # 2. 权重和约束
        if stage['weight_sum'] is not None:
            constraints.append(cp.sum(w) == stage['weight_sum'])
        
        # 3. 范数约束
        if stage['norm_bound'] is not None:
            constraints.append(cp.norm(w, 1) <= stage['norm_bound'])
        
        # 4. 换手率约束
        if stage['turnover_bound'] is not None and current_weights is not None:
            aligned_current = current_weights.reindex(factor_scores.index, fill_value=0.0)
            constraints.append(cp.norm(w - aligned_current.values, 1) <= stage['turnover_bound'])
        
        # 5. 资金费率约束（如果开启）
        if self.enable_funding_rate_constraint:
            constraints.append(w @ funding_rates.values >= 0)
        
        problem = cp.Problem(objective, constraints)
        return problem, w