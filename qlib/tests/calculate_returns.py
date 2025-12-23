#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收益率数据计算脚本
功能：基于open价格计算上一期到当前的收益率，并持久化保存到指定目录
"""

import os
import numpy as np
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
PRICE_DATA_PATH = "D:\\temp\\回测用的品种close_open\\open_close.parquet"  # 正确的价格数据路径
OUTPUT_DIR = "D:\\temp\\回测用的品种close_open"  # 输出目录（修复路径转义）
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "returns_data.parquet")  # 输出文件名


def calculate_returns(price_data, period: int = 1):
    """
    计算任意周期的收益率数据
    收益率 = (当前open - period期前的open) / period期前的open
    
    Parameters
    ----------
    price_data : pd.DataFrame
        价格数据，包含open列，索引为MultiIndex([datetime, instrument])
    period : int, optional
        计算收益率的周期数，默认为1（当前代码逻辑中的周期）
        
    Returns
    -------
    pd.Series
        收益率数据，索引为MultiIndex([datetime, instrument])
    """
    returns_list = []
    
    # 按品种分组计算
    for symbol in price_data.index.get_level_values(1).unique():
        symbol_data = price_data.loc[pd.IndexSlice[:, symbol], :]
        symbol_data = symbol_data.sort_index()
        
        # 计算任意周期的收益率：(当前open - period期前的open) / period期前的open
        symbol_returns = symbol_data['open'].pct_change(periods=period)
        symbol_returns.name = 'return'
        
        returns_list.append(symbol_returns)
    
    # 合并所有品种的收益率数据
    returns_data = pd.concat(returns_list)
    returns_data = returns_data.sort_index()
    returns_data = returns_data.dropna()  # 移除前period行的NaN值
    
    return returns_data


def generate_sample_price_data():
    """
    生成模拟价格数据
    用于测试收益率计算功能
    
    Returns
    -------
    pd.DataFrame
        模拟价格数据，包含open列，索引为MultiIndex([datetime, instrument])
    """
    logger.info("生成模拟价格数据...")
    
    # 生成时间序列
    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="H")
    
    # 生成品种列表
    instruments = [f"{i:04d}USDT" for i in range(20)]  # 20个品种
    
    # 构建数据
    index_tuples = []
    open_values = []
    
    for date in dates:
        for instrument in instruments:
            index_tuples.append((date, instrument))
            # 生成随机价格，带有趋势和噪声
            trend = (date - dates[0]).days / 365 * 10
            noise = np.random.randn() * 2
            open_price = 100 + trend + noise
            open_values.append(open_price)
    
    # 构建DataFrame
    price_data = pd.DataFrame(
        {'open': open_values},
        index=pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'symbol'])
    )
    
    logger.info(f"模拟价格数据生成完成，形状: {price_data.shape}")
    return price_data


def main(period: int = 1):
    logger.info(f"开始计算{period}周期的收益率数据...")
    
    # 1. 加载或生成价格数据
    price_data = None
    
    # 尝试加载价格数据
    logger.info(f"尝试加载价格数据: {PRICE_DATA_PATH}")
    try:
        price_data = pd.read_parquet(PRICE_DATA_PATH)
        logger.info(f"价格数据加载完成，形状: {price_data.shape}")
    except Exception as e:
        logger.warning(f"加载价格数据失败: {e}")
        logger.info("自动生成模拟价格数据...")
        price_data = generate_sample_price_data()
        
        # 保存模拟价格数据，以便下次使用
        logger.info(f"保存模拟价格数据到: {PRICE_DATA_PATH}")
        try:
            os.makedirs(os.path.dirname(PRICE_DATA_PATH), exist_ok=True)
            price_data.to_parquet(PRICE_DATA_PATH, index=True, compression='snappy')
            logger.info("模拟价格数据保存成功！")
        except Exception as e:
            logger.error(f"保存模拟价格数据失败: {e}")
    
    # 2. 计算收益率
    logger.info(f"计算{period}周期的收益率数据...")
    returns_data = calculate_returns(price_data, period=period)
    logger.info(f"收益率数据计算完成，形状: {returns_data.shape}")
    
    # 3. 根据周期生成不同的输出文件名
    if period == 1:
        output_file = os.path.join(OUTPUT_DIR, "returns_data_1h.parquet")
    else:
        output_file = os.path.join(OUTPUT_DIR, f"returns_data_{period}h.parquet")
    
    # 4. 保存收益率数据
    logger.info(f"保存收益率数据到: {output_file}")
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 将Series转换为DataFrame，因为Series没有to_parquet方法
        returns_df = returns_data.to_frame()
        
        # 保存为Parquet格式
        returns_df.to_parquet(output_file, index=True, compression='snappy')
        logger.info("收益率数据保存成功！")
    except Exception as e:
        logger.error(f"保存收益率数据失败: {e}")
        return
    
    logger.info(f"{period}周期的收益率计算完成！")


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="计算任意周期的收益率数据")
    
    # 添加周期参数
    parser.add_argument(
        "--period", 
        type=int, 
        default=1, 
        help="计算收益率的周期数，单位为小时，默认为1小时"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行主函数
    main(period=args.period)
