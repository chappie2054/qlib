# gp_factor_search.py
import time
import math
import logging
import random
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from deap import base, creator, gp, tools
from joblib import Parallel, delayed, parallel_backend
import operator
import multiprocessing

# 导入tqdm用于显示进度条
try:
    # 尝试导入notebook版本的tqdm
    from tqdm.notebook import tqdm
    TQDM_NOTEBOOK_AVAILABLE = True
except ImportError:
    # 如果notebook版本不可用，使用普通版本
    from tqdm import tqdm
    TQDM_NOTEBOOK_AVAILABLE = False

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GP-Factor")

# Import custom operators
from .ops import *

# -------------------------
# User settings
# -------------------------
# Time windows (explicit)
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END   = pd.Timestamp("2023-12-12")
VALID_START = pd.Timestamp("2023-12-13")
VALID_END   = pd.Timestamp("2024-07-05")
TEST_START  = pd.Timestamp("2024-07-06")
TEST_END    = pd.Timestamp("2025-08-25")

# windows allowed for rolling operators
WINDOWS = [3, 5, 10, 20, 40, 60]

# complexity penalty coefficient
COMPLEXITY_PENALTY = 0.002

# joblib n_jobs recommendation (will default to CPU-2)
# DEFAULT_N_JOBS = max(1, multiprocessing.cpu_count() - 2)
DEFAULT_N_JOBS = max(1, multiprocessing.cpu_count() - 5)

# Genetic params
POP_SIZE = 1000
NGEN = 3
CX_PB = 0.4
MUT_PB = 1
TOURNSIZE = 20
HOF_SIZE = 6

# -------------------------
# Data preparation function
# -------------------------
def prepare_data(old_data_swap, train_start=TRAIN_START, train_end=TRAIN_END, 
                 valid_start=VALID_START, valid_end=VALID_END,
                 test_start=TEST_START, test_end=TEST_END):
    """
    准备训练、验证和测试数据
    
    参数:
        old_data_swap: 原始数据DataFrame
        train_start, train_end: 训练集时间范围
        valid_start, valid_end: 验证集时间范围
        test_start, test_end: 测试集时间范围
        
    返回:
        train_val_df: 训练+验证数据
        test_df: 测试数据
    """
    data_all = old_data_swap.copy()
    
    # Ensure index names
    if data_all.index.names[0] is None or data_all.index.names[1] is None:
        data_all.index.set_names(["datetime", "instrument"], inplace=True)
    logger.info("数据载入完成。样本数: %d，columns: %s", len(data_all), list(data_all.columns))
    logger.info("Index names: %s", data_all.index.names)
    
    # Build label: Ref(close, -10) / Ref(open, -1) - 1
    logger.info("构造标签 ...")
    t0 = time.perf_counter()
    close_shift_10 = data_all["close"].groupby(level="instrument").shift(-10)
    open_shift_1 = data_all["open"].groupby(level="instrument").shift(-1)
    data_all["label_raw"] = close_shift_10 / open_shift_1 - 1.0
    logger.info("标签构造耗时: %.3fs", time.perf_counter() - t0)
    
    # Construct train+val index for fitness
    train_val_mask = ((data_all.index.get_level_values(0) >= train_start) &
                      (data_all.index.get_level_values(0) <= valid_end))
    train_val_df = data_all.loc[train_val_mask].copy()
    logger.info("Train+Val 样本数: %d (从 %s 到 %s)", len(train_val_df), train_start.date(), valid_end.date())
    
    # also keep full test for later evaluation
    test_mask = ((data_all.index.get_level_values(0) >= test_start) &
                 (data_all.index.get_level_values(0) <= test_end))
    test_df = data_all.loc[test_mask].copy()
    logger.info("Test 样本数: %d (从 %s 到 %s)", len(test_df), test_start.date(), test_end.date())
    
    return train_val_df, test_df

# -------------------------
# Helper numeric-safe functions that operate on pandas Series (MultiIndex aligned)
# Each operator accepts Series (index MultiIndex) or two Series, and returns Series.
# We implement per-instrument rolling/time ops using groupby('instrument')
# -------------------------
# All operators have been moved to ops.py module

def op_signedpower(x, a):
    return x.apply(lambda v: math.copysign((abs(v) ** a) if pd.notna(v) else np.nan, v))

# All operators have been moved to ops.py module

# -------------------------
# Build DEAP PrimitiveSet with these operators
# We design primitives to accept pandas.Series and ints for window params.
# DEAP types: we will use Typed GP to specify type safety: use object type mapping
# But for simplicity we use untyped gp.PrimitiveSet and ensure runtime args are correct.
# -------------------------
pset = gp.PrimitiveSet("MAIN", 5)  # args: close, open, high, low, volume
pset.renameArguments(ARG0='close', ARG1='open', ARG2='high', ARG3='low', ARG4='volume')

# add math primitives (Series->Series or Series,Series->Series)
pset.addPrimitive(op_add, 2)
pset.addPrimitive(op_sub, 2)
pset.addPrimitive(op_mul, 2)
pset.addPrimitive(op_div, 2)
pset.addPrimitive(op_abs, 1)
pset.addPrimitive(op_sqrt, 1)
pset.addPrimitive(op_log, 1)
pset.addPrimitive(op_inv, 1)

# custom primitives
pset.addPrimitive(op_rank, 1)
pset.addPrimitive(op_delay, 2)           # (Series, int)
pset.addPrimitive(op_correlation, 3)     # (X, Y, d)
pset.addPrimitive(op_covariance, 3)
pset.addPrimitive(op_scale, 2)           # (X, a)
pset.addPrimitive(op_delta, 2)
pset.addPrimitive(op_signedpower, 2)
pset.addPrimitive(op_decay_linear, 2)
# pset.addPrimitive(op_indneutralize, 2)   # (X, indclass) -> placeholder
pset.addPrimitive(op_ts_min, 2)
pset.addPrimitive(op_ts_max, 2)
pset.addPrimitive(op_ts_argmin, 2)
pset.addPrimitive(op_ts_argmax, 2)
pset.addPrimitive(op_ts_rank, 2)
pset.addPrimitive(op_ts_sum, 2)
pset.addPrimitive(op_ts_product, 2)
pset.addPrimitive(op_ts_stddev, 2)

# Terminals: none bound to actual series here; compile returns function expecting 5 series inputs
# Add ephemeral constants for numeric constants and integer terminals for windows
for w in WINDOWS:
    # add as ephemeral constant returning int -> need to register as terminal: use name const_w
    pset.addTerminal(w)

# floating constants
pset.addEphemeralConstant("const_rand", lambda: random.uniform(-1.0, 1.0))

logger.info("PrimitiveSet 构建完毕。primitives=%d, terminals=%d",
            sum(len(v) for v in pset.primitives.values()),
            sum(len(v) for v in pset.terminals.values()))

# -------------------------
# DEAP toolbox & individual types
# -------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# register genetic operators
toolbox.register("mate", gp.cxOnePoint)
# subtree mutation
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mut_subtree", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# node replacement
toolbox.register("mut_node", gp.mutNodeReplacement, pset=pset)
# hoist mutation
toolbox.register("mut_hoist", gp.mutShrink)

# -------------------------
# mixed mutation with custom probabilities
# -------------------------
def mixed_mutation_custom(individual):
    # 按你的要求：Hoist=0, Subtree=0.01, Node=0.01, 每个节点Node变异=0.4
    r = random.random()
    if r < 0.01:  # 子树变异
        mutant, = toolbox.mut_subtree(individual)
        return mutant,
    elif r < 0.02:  # 点变异
        mutant, = toolbox.mut_node(individual)
        return mutant,
    elif r < 0.42:  # 点变异中父代每个节点进行变异
        mutant, = gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
        return mutant,
    else:
        return individual,  # 不变 (包括 Hoist=0)

toolbox.register("mutate", mixed_mutation_custom)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

# limit size to prevent bloat
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=120))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=120))

# -------------------------
# fitness evaluation: compile individual -> function -> compute factor Series on train+val,
# preprocess factor, compute RankIC mean across dates
# -------------------------
def evaluate_individual_return_metrics(individual, df_trainval):
    """
    Returns: fitness (float), metrics dict (mean_ic, std_ic, n_days, complexity)
    """
    expr_str = str(individual)
    try:
        func = toolbox.compile(expr=individual)
    except Exception as e:
        logger.debug("compile failed for %s: %s", expr_str, e)
        return -1e6, {"mean_ic": float("nan"), "std_ic": float("nan"), "n_days": 0, "complexity": len(individual)}

    # execute compiled function on Series inputs
    try:
        # pass the full train+val columns as Series (MultiIndex aligned)
        close = df_trainval["close"]
        open_ = df_trainval["open"]
        high = df_trainval["high"]
        low = df_trainval["low"]
        vol = df_trainval["volume"]

        t0 = time.time()
        # function is expected to operate on pandas.Series
        factor_ser = func(close, open_, high, low, vol)
        logger.debug("expr %s 执行耗时 %.4f 秒", expr_str, time.time() - t0)

        # if scalar return, broadcast
        if np.isscalar(factor_ser):
            factor_ser = pd.Series(float(factor_ser), index=df_trainval.index)

        # ensure aligned index
        if not isinstance(factor_ser, pd.Series):
            # try convert
            factor_ser = pd.Series(factor_ser, index=df_trainval.index)

    except Exception as e:
        logger.debug("execution failed for %s: %s", expr_str, e)
        return -1e6, {"mean_ic": float("nan"), "std_ic": float("nan"), "n_days": 0, "complexity": len(individual)}

    # Preprocess factor: per-date MAD winsorize + zscore
    try:
        t1 = time.time()
        factor_proc = preprocess_mad_zscore(factor_ser)
        logger.debug("expr %s 预处理耗时 %.4f 秒", expr_str, time.time() - t1)
    except Exception as e:
        logger.debug("preprocess failed for %s: %s", expr_str, e)
        return -1e6, {"mean_ic": float("nan"), "std_ic": float("nan"), "n_days": 0, "complexity": len(individual)}

    # label: use raw label and rank per date
    label_raw = df_trainval["label_raw"]
    # we keep label rank as cross-sectional pct
    # compute mean RankIC
    t2 = time.time()
    mean_ic, ic_arr = rank_ic_mean(factor_proc, label_raw)
    logger.debug("expr %s 计算 RankIC 耗时 %.4f 秒", expr_str, time.time() - t2)
    if np.isnan(mean_ic):
        # no valid days
        return -1e6, {"mean_ic": float("nan"), "std_ic": float("nan"), "n_days": 0, "complexity": len(individual)}

    std_ic = float(np.nanstd(ic_arr))
    n_days = len(ic_arr)
    
    # 计算总天数
    total_days = df_trainval.index.get_level_values('datetime').nunique()
    
    # 如果有效天数占比不足80%，则认为该因子不可靠，给予很低的适应度
    if n_days / total_days < 0.7:
        logger.debug("expr %s 有效天数占比不足 (n_days=%d, total_days=%d, ratio=%.2f < 0.7)，适应度设为-1e6", expr_str, n_days, total_days, n_days / total_days)
        return -1e6, {"mean_ic": float("nan"), "std_ic": float("nan"), "n_days": n_days, "complexity": len(individual)}
    
    # fitness is mean_ic (could subtract std or complexity)
    fitness = mean_ic - 0.0 * std_ic - COMPLEXITY_PENALTY * len(individual)

    metrics = {"mean_ic": mean_ic, "std_ic": std_ic, "n_days": n_days, "complexity": len(individual)}
    return float(fitness), metrics

# wrapper used for parallel calls: returns (fitness, metrics, expr_string)
def _eval_individual_for_parallel(ind, df_trainval):
    f, metrics = evaluate_individual_return_metrics(ind, df_trainval)
    return (f, metrics, str(ind))

# -------------------------
# Main GP loop with joblib parallel evaluation
# -------------------------
def run_gp_and_save(df_trainval, test_df=None, pop_size=POP_SIZE, ngen=NGEN, cxpb=CX_PB, mutpb=MUT_PB, n_jobs=DEFAULT_N_JOBS):
    logger.info("准备启动 GP: pop=%d ngen=%d cxpb=%.2f mutpb=%.2f n_jobs=%d", pop_size, ngen, cxpb, mutpb, n_jobs)

    # init population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(HOF_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda vs: float(np.mean([v[0] for v in vs])))
    stats.register("max", lambda vs: float(np.max([v[0] for v in vs])))

    # initial evaluation
    logger.info("初始种群评估（并行）...")
    with parallel_backend("loky", n_jobs=n_jobs):
        results = Parallel(n_jobs=n_jobs, temp_folder=r"D:\temp_joblib", verbose=10)(
            delayed(_eval_individual_for_parallel)(ind, df_trainval) for ind in tqdm(pop, desc="初始种群评估")
        )
    # assign fitness and optionally log metrics
    for ind, (fit, metrics, expr) in zip(pop, results):
        ind.fitness.values = (fit,)
    logger.info("初代评估完成: avg=%.6f, max=%.6f", stats.compile(pop)["avg"], stats.compile(pop)["max"])

    # evolution
    for gen in range(1, ngen + 1):
        gen_start = time.perf_counter()
        logger.info("=== Generation %d ===", gen)
        # selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # mutate (mixed mutation)
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # evaluate invalid individuals in parallel
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        logger.info("Generation %d: 待评估个体数 %d", gen, len(invalid))
        if invalid:
            with parallel_backend("loky", n_jobs=n_jobs):
                results = Parallel(n_jobs=n_jobs, temp_folder=r"D:\temp_joblib", verbose=10)(
                    delayed(_eval_individual_for_parallel)(ind, df_trainval) for ind in tqdm(invalid, desc=f"第{gen}代个体评估")
                )
            # assign
            for ind, (fit, metrics, expr) in zip(invalid, results):
                ind.fitness.values = (fit,)

        # replace population
        pop[:] = offspring
        hof.update(pop)

        rec = stats.compile(pop)
        logger.info("Generation %d done: avg=%.6f, max=%.6f, elapsed=%.2fs", gen, rec["avg"], rec["max"], time.perf_counter() - gen_start)

        # optional: save intermediate hof or logs every few generations
        if gen % 10 == 0 or gen == ngen:
            # save top k from hof to csv
            rows = []
            for i, ind in enumerate(hof):
                f, metrics = evaluate_individual_return_metrics(ind, df_trainval)
                rows.append({
                    "rank": i+1, "expr": str(ind), "fitness": f,
                    "mean_ic": metrics.get("mean_ic"), "std_ic": metrics.get("std_ic"), "n_days": metrics.get("n_days"),
                    "complexity": metrics.get("complexity")
                })
            pd.DataFrame(rows).to_csv(f"gp_hof_gen{gen}.csv", index=False)
            logger.info("Saved hof snapshot to gp_hof_gen%d.csv", gen)

    # after evolution, evaluate hof on train+val and test for reporting
    logger.info("进化结束，评估 HallOfFame 上的个体（train+val & test）...")
    final_rows = []
    for i, ind in enumerate(hof):
        f_trainval, m_trainval = evaluate_individual_return_metrics(ind, df_trainval)
        # 如果提供了测试数据，则在测试集上评估，否则测试集指标设为None
        if test_df is not None:
            f_test, m_test = evaluate_individual_return_metrics(ind, test_df)
        else:
            f_test, m_test = None, {"mean_ic": None, "std_ic": None, "n_days": None}
        final_rows.append({
            "rank": i+1,
            "expr": str(ind),
            "fitness_trainval": f_trainval,
            "mean_ic_trainval": m_trainval.get("mean_ic"),
            "std_ic_trainval": m_trainval.get("std_ic"),
            "n_days_trainval": m_trainval.get("n_days"),
            "complexity": m_trainval.get("complexity"),
            "fitness_test": f_test,
            "mean_ic_test": m_test.get("mean_ic"),
            "std_ic_test": m_test.get("std_ic"),
            "n_days_test": m_test.get("n_days"),
        })
    outdf = pd.DataFrame(final_rows)
    outdf.to_csv("gp_final_hof.csv", index=False)
    logger.info("最终结果已保存到 gp_final_hof.csv")
    return hof, outdf

# -------------------------
# Run (example)
# -------------------------
if __name__ == "__main__":
    # 仅在直接运行脚本时执行数据加载
    if "old_data_swap" not in globals():
        raise RuntimeError("请在运行前把你的 DataFrame 变量命名为 old_data_swap 并加载到环境中。")
    
    random.seed(42)
    np.random.seed(42)
    
    # 准备数据
    train_val_df, test_df = prepare_data(old_data_swap)
    
    # 运行GP搜索
    hof, outdf = run_gp_and_save(train_val_df, test_df, pop_size=POP_SIZE, ngen=NGEN, cxpb=CX_PB, mutpb=MUT_PB, n_jobs=DEFAULT_N_JOBS)
    logger.info("Top expressions saved. Top1: %s", str(hof[0]) if len(hof) > 0 else "None")
