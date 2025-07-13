# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import warnings
import numpy as np
import pandas as pd
import bisect
from qlib.utils.data import guess_horizon
from qlib.utils import init_instance_by_config
from copy import deepcopy
from scipy.stats import spearmanr
from typing import Callable, Union, List, Tuple, Dict, Text, Optional
from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc


device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=torch.float, device=device)  # pylint: disable=E1101
    return x


def _create_ts_slices(index, seq_len):
    """
    create time series slices from pandas index

    Args:
        index (pd.MultiIndex): pandas multiindex with <instrument, datetime> order
        seq_len (int): sequence length
    """
    assert isinstance(index, pd.MultiIndex), "unsupported index type"
    assert seq_len > 0, "sequence length should be larger than 0"
    assert index.is_monotonic_increasing, "index should be sorted"

    # number of dates for each instrument
    sample_count_by_insts = index.to_series().groupby(level=0, group_keys=False).size().values

    # start index for each instrument
    start_index_of_insts = np.roll(np.cumsum(sample_count_by_insts), 1)
    start_index_of_insts[0] = 0

    # all the [start, stop) indices of features
    # features between [start, stop) will be used to predict label at `stop - 1`
    slices = []
    for cur_loc, cur_cnt in zip(start_index_of_insts, sample_count_by_insts):
        for stop in range(1, cur_cnt + 1):
            end = cur_loc + stop
            start = max(end - seq_len, 0)
            slices.append(slice(start, end))
    slices = np.array(slices, dtype="object")

    assert len(slices) == len(index)  # the i-th slice = index[i]

    return slices


def _get_date_parse_fn(target):
    """get date parse function

    This method is used to parse date arguments as target type.

    Example:
        get_date_parse_fn('20120101')('2017-01-01') => '20170101'
        get_date_parse_fn(20120101)('2017-01-01') => 20170101
    """
    if isinstance(target, int):

        def _fn(x):
            return int(str(x).replace("-", "")[:8])  # 20200201

    elif isinstance(target, str) and len(target) == 8:

        def _fn(x):
            return str(x).replace("-", "")[:8]  # '20200201'

    else:

        def _fn(x):
            return x  # '2021-01-01'

    return _fn


def _maybe_padding(x, seq_len, zeros=None):
    """padding 2d <time * feature> data with zeros

    Args:
        x (np.ndarray): 2d data with shape <time * feature>
        seq_len (int): target sequence length
        zeros (np.ndarray): zeros with shape <seq_len * feature>
    """
    assert seq_len > 0, "sequence length should be larger than 0"
    if zeros is None:
        zeros = np.zeros((seq_len, x.shape[1]), dtype=np.float32)
    else:
        assert len(zeros) >= seq_len, "zeros matrix is not large enough for padding"
    if len(x) != seq_len:  # padding zeros
        x = np.concatenate([zeros[: seq_len - len(x), : x.shape[1]], x], axis=0)
    return x


class MTSDatasetH(DatasetH):
    """Memory Augmented Time Series Dataset

    Args:
        handler (DataHandler): data handler
        segments (dict): data split segments
        seq_len (int): time series sequence length
        horizon (int): label horizon
        num_states (int): how many memory states to be added
        memory_mode (str): memory mode (daily or sample)
        batch_size (int): batch size (<0 will use daily sampling)
        n_samples (int): number of samples in the same day
        shuffle (bool): whether shuffle data
        drop_last (bool): whether drop last batch < batch_size
        input_size (int): reshape flatten rows as this input_size (backward compatibility)
    """

    def __init__(
        self,
        handler,
        segments,
        seq_len=60,
        horizon=0,
        num_states=0,
        memory_mode="sample",
        batch_size=-1,
        n_samples=None,
        shuffle=True,
        drop_last=False,
        input_size=None,
        **kwargs,
    ):
        if horizon == 0:
            # Try to guess horizon
            if isinstance(handler, (dict, str)):
                handler = init_instance_by_config(handler)
            assert "label" in getattr(handler.data_loader, "fields", None)
            label = handler.data_loader.fields["label"][0][0]
            horizon = guess_horizon([label])

        assert num_states == 0 or horizon > 0, "please specify `horizon` to avoid data leakage"
        assert memory_mode in ["sample", "daily"], "unsupported memory mode"
        assert memory_mode == "sample" or batch_size < 0, "daily memory requires daily sampling (`batch_size < 0`)"
        assert batch_size != 0, "invalid batch size"

        if batch_size > 0 and n_samples is not None:
            warnings.warn("`n_samples` can only be used for daily sampling (`batch_size < 0`)")

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_states = num_states
        self.memory_mode = memory_mode
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.input_size = input_size
        self.params = (batch_size, n_samples, drop_last, shuffle)  # for train/eval switch

        super().__init__(handler, segments, **kwargs)

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        super().setup_data(**kwargs)

        if handler_kwargs is not None:
            self.handler.setup_data(**handler_kwargs)

        # pre-fetch data and change index to <code, date>
        # NOTE: we will use inplace sort to reduce memory use
        try:
            df = self.handler._learn.copy()  # use copy otherwise recorder will fail
            # FIXME: currently we cannot support switching from `_learn` to `_infer` for inference
        except Exception:
            warnings.warn("cannot access `_learn`, will load raw data")
            df = self.handler._data.copy()
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        # convert to numpy
        self._data = df["feature"].values.astype("float32")
        np.nan_to_num(self._data, copy=False)  # NOTE: fillna in case users forget using the fillna processor
        self._label = df["label"].squeeze().values.astype("float32")
        self._index = df.index

        if self.input_size is not None and self.input_size != self._data.shape[1]:
            warnings.warn("the data has different shape from input_size and the data will be reshaped")
            assert self._data.shape[1] % self.input_size == 0, "data mismatch, please check `input_size`"

        # create batch slices
        self._batch_slices = _create_ts_slices(self._index, self.seq_len)

        # create daily slices
        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}  # sorted by date
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append(self._batch_slices[i])
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(list(daily_slices.keys()))  # index is the original date index

        # add memory (sample wise and daily)
        if self.memory_mode == "sample":
            self._memory = np.zeros((len(self._data), self.num_states), dtype=np.float32)
        elif self.memory_mode == "daily":
            self._memory = np.zeros((len(self._daily_index), self.num_states), dtype=np.float32)
        else:
            raise ValueError(f"invalid memory_mode `{self.memory_mode}`")

        # padding tensor
        self._zeros = np.zeros((self.seq_len, max(self.num_states, self._data.shape[1])), dtype=np.float32)

    def _prepare_seg(self, slc, **kwargs):
        fn = _get_date_parse_fn(self._index[0][1])
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError(f"This type of input is not supported")
        start_date = pd.Timestamp(fn(start))
        end_date = pd.Timestamp(fn(stop))
        obj = copy.copy(self)  # shallow copy
        # NOTE: Seriable will disable copy `self._data` so we manually assign them here
        obj._data = self._data  # reference (no copy)
        obj._label = self._label
        obj._index = self._index
        obj._memory = self._memory
        obj._zeros = self._zeros
        # update index for this batch
        date_index = self._index.get_level_values(1)
        obj._batch_slices = self._batch_slices[(date_index >= start_date) & (date_index <= end_date)]
        mask = (self._daily_index.values >= start_date) & (self._daily_index.values <= end_date)
        obj._daily_slices = self._daily_slices[mask]
        obj._daily_index = self._daily_index[mask]
        return obj

    def restore_index(self, index):
        return self._index[index]

    def restore_daily_index(self, daily_index):
        return pd.Index(self._daily_index.loc[daily_index])

    def assign_data(self, index, vals):
        if self.num_states == 0:
            raise ValueError("cannot assign data as `num_states==0`")
        if isinstance(vals, torch.Tensor):
            vals = vals.detach().cpu().numpy()
        self._memory[index] = vals

    def clear_memory(self):
        if self.num_states == 0:
            raise ValueError("cannot clear memory as `num_states==0`")
        self._memory[:] = 0

    def train(self):
        """enable traning mode"""
        self.batch_size, self.n_samples, self.drop_last, self.shuffle = self.params

    def eval(self):
        """enable evaluation mode"""
        self.batch_size = -1
        self.n_samples = None
        self.drop_last = False
        self.shuffle = False

    def _get_slices(self):
        if self.batch_size < 0:  # daily sampling
            slices = self._daily_slices.copy()
            batch_size = -1 * self.batch_size
        else:  # normal sampling
            slices = self._batch_slices.copy()
            batch_size = self.batch_size
        return slices, batch_size

    def __len__(self):
        slices, batch_size = self._get_slices()
        if self.drop_last:
            return len(slices) // batch_size
        return (len(slices) + batch_size - 1) // batch_size

    def __iter__(self):
        slices, batch_size = self._get_slices()
        indices = np.arange(len(slices))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(indices))[::batch_size]:
            if self.drop_last and i + batch_size > len(indices):
                break

            data = []  # store features
            label = []  # store labels
            index = []  # store index
            state = []  # store memory states
            daily_index = []  # store daily index
            daily_count = []  # store number of samples for each day

            for j in indices[i : i + batch_size]:
                # normal sampling: self.batch_size > 0 => slices is a list => slices_subset is a slice
                # daily sampling: self.batch_size < 0 => slices is a nested list => slices_subset is a list
                slices_subset = slices[j]

                # daily sampling
                # each slices_subset contains a list of slices for multiple stocks
                # NOTE: daily sampling is used in 1) eval mode, 2) train mode with self.batch_size < 0
                if self.batch_size < 0:
                    # store daily index
                    idx = self._daily_index.index[j]  # daily_index.index is the index of the original data
                    daily_index.append(idx)

                    # store daily memory if specified
                    # NOTE: daily memory always requires daily sampling (self.batch_size < 0)
                    if self.memory_mode == "daily":
                        slc = slice(max(idx - self.seq_len - self.horizon, 0), max(idx - self.horizon, 0))
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros))

                    # down-sample stocks and store count
                    if self.n_samples and 0 < self.n_samples < len(slices_subset):  # intraday subsample
                        slices_subset = np.random.choice(slices_subset, self.n_samples, replace=False)
                    daily_count.append(len(slices_subset))

                # normal sampling
                # each slices_subset is a single slice
                # NOTE: normal sampling is used in train mode with self.batch_size > 0
                else:
                    slices_subset = [slices_subset]

                for slc in slices_subset:
                    # legacy support for Alpha360 data by `input_size`
                    if self.input_size:
                        data.append(self._data[slc.stop - 1].reshape(self.input_size, -1).T)
                    else:
                        data.append(_maybe_padding(self._data[slc], self.seq_len, self._zeros))

                    if self.memory_mode == "sample":
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros)[: -self.horizon])

                    label.append(self._label[slc.stop - 1])
                    index.append(slc.stop - 1)

                    # end slices loop

                # end indices batch loop

            # concate
            data = _to_tensor(np.stack(data))
            state = _to_tensor(np.stack(state))
            label = _to_tensor(np.stack(label))
            index = np.array(index)
            daily_index = np.array(daily_index)
            daily_count = np.array(daily_count)

            # yield -> generator
            yield {
                "data": data,
                "label": label,
                "state": state,
                "index": index,
                "daily_index": daily_index,
                "daily_count": daily_count,
            }

        # end indice loop


###################################################################################
# lqa: for MASTER
# class marketDataHandler(DataHandlerLP):
#     """Market Data Handler for MASTER (see `examples/benchmarks/MASTER`)
#
#     Args:
#         instruments (str): instrument list
#         start_time (str): start time
#         end_time (str): end time
#         freq (str): data frequency
#         infer_processors (list): inference processors
#         learn_processors (list): learning processors
#         fit_start_time (str): fit start time
#         fit_end_time (str): fit end time
#         process_type (str): process type
#         filter_pipe (list): filter pipe
#         inst_processors (list): instrument processors
#     """
#
#     def __init__(
#             self,
#             instruments="csi300",
#             start_time=None,
#             end_time=None,
#             freq="day",
#             infer_processors=[],
#             learn_processors=[],
#             fit_start_time=None,
#             fit_end_time=None,
#             process_type=DataHandlerLP.PTYPE_A,
#             filter_pipe=None,
#             inst_processors=None,
#             **kwargs
#     ):
#         infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
#         learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
#
#         data_loader = {
#             "class": "QlibDataLoader",
#             "kwargs": {
#                 "config": {
#                     "feature": self.get_feature_config(),
#                 },
#                 "filter_pipe": filter_pipe,
#                 "freq": freq,
#                 "inst_processors": inst_processors,
#             },
#         }
#         super().__init__(
#             instruments=instruments,
#             start_time=start_time,
#             end_time=end_time,
#             data_loader=data_loader,
#             infer_processors=infer_processors,
#             learn_processors=learn_processors,
#             process_type=process_type,
#             **kwargs
#         )
#
#     @staticmethod
#     def get_feature_config():
#         """
#         Get market feature (63-dimensional), which are csi100 index, csi300 index, csi500 index.
#         The first list is the name to be shown for the feature, and the second list is the feature to fecth.
#         """
#         return (
#             ['Mask($close/Ref($close,1)-1, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,5), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,5), "MARKET")', 'Mask(Mean($volume,5)/$volume, "MARKET")',
#              'Mask(Std($volume,5)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,10), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,10), "MARKET")', 'Mask(Mean($volume,10)/$volume, "MARKET")',
#              'Mask(Std($volume,10)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,20), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,20), "MARKET")', 'Mask(Mean($volume,20)/$volume, "MARKET")',
#              'Mask(Std($volume,20)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,30), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,30), "MARKET")', 'Mask(Mean($volume,30)/$volume, "MARKET")',
#              'Mask(Std($volume,30)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,60), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,60), "MARKET")', 'Mask(Mean($volume,60)/$volume, "MARKET")',
#              'Mask(Std($volume,60)/$volume, "MARKET")'],
#             ['Mask($close/Ref($close,1)-1, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,5), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,5), "MARKET")', 'Mask(Mean($volume,5)/$volume, "MARKET")',
#              'Mask(Std($volume,5)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,10), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,10), "MARKET")', 'Mask(Mean($volume,10)/$volume, "MARKET")',
#              'Mask(Std($volume,10)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,20), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,20), "MARKET")', 'Mask(Mean($volume,20)/$volume, "MARKET")',
#              'Mask(Std($volume,20)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,30), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,30), "MARKET")', 'Mask(Mean($volume,30)/$volume, "MARKET")',
#              'Mask(Std($volume,30)/$volume, "MARKET")', 'Mask(Mean($close/Ref($close,1)-1,60), "MARKET")',
#              'Mask(Std($close/Ref($close,1)-1,60), "MARKET")', 'Mask(Mean($volume,60)/$volume, "MARKET")',
#              'Mask(Std($volume,60)/$volume, "MARKET")']
#         )
class marketDataHandler(DataHandlerLP):
    def __init__(
            self,
            instruments="csi300",
            start_time=None,
            end_time=None,
            freq="1min",  # 默认1分钟
            infer_processors=[],
            learn_processors=[],
            fit_start_time=None,
            fit_end_time=None,
            process_type=DataHandlerLP.PTYPE_A,
            filter_pipe=None,
            inst_processors=None,
            **kwargs
    ):
        self.freq = freq  # 保存频率，供get_feature_config使用

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        """
        根据freq动态生成特征表达式
        """

        # 定义时间单位到分钟的换算（默认1min对应1）
        freq_map = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
            "60min": 60,
            "day": 1440,
            "week": 10080,
            "month": 43200,
        }

        # 获取单位分钟数（默认1）
        unit_min = freq_map.get(self.freq, 1)

        # 定义窗口长度（天为单位）
        windows_day = [1, 5, 10, 20, 30, 60]

        # 换算窗口长度为对应频率的步数（举例1小时=60分钟）
        windows_steps = [int(day * 1440 / unit_min) for day in windows_day]

        feature_names = []
        feature_exprs = []

        # 计算收益率时，用offset为1个freq单位（即前一个bar）
        ref_offset = 1

        # 基础收益率特征（前一个bar的收益率）

        for w in windows_steps:
            feature_names.append(f'Mask($close/Ref($close,{w})-1, "MARKET")')
            feature_names.append(f'Mask(Mean($close/Ref($close,{w})-1,{w}), "MARKET")')
            feature_names.append(f'Mask(Std($close/Ref($close,{w})-1,{w}), "MARKET")')
            feature_names.append(f'Mask(Mean($volume,{w})/$volume, "MARKET")')
            feature_names.append(f'Mask(Std($volume,{w})/$volume, "MARKET")')

            feature_exprs.append(f'Mask($close/Ref($close,{w})-1, "MARKET")')
            feature_exprs.append(f'Mask(Mean($close/Ref($close,{w})-1,{w}), "MARKET")')
            feature_exprs.append(f'Mask(Std($close/Ref($close,{w})-1,{w}), "MARKET")')
            feature_exprs.append(f'Mask(Mean($volume,{w})/$volume, "MARKET")')
            feature_exprs.append(f'Mask(Std($volume,{w})/$volume, "MARKET")')

        return feature_names, feature_exprs


class MASTERTSDatasetH(TSDatasetH):
    """
    MASTER Time Series Dataset with Handler

    Args:
        market_data_handler_config (dict): market data handler config
    """

    def __init__(
            self,
            market_data_handler_config=Dict,
            **kwargs,
    ):
        super().__init__(**kwargs)
        marketdl = marketDataHandler(**market_data_handler_config)
        self.market_dataset = DatasetH(marketdl, segments=self.segments)

    def get_market_information(
            self,
            slc: slice,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        return self.market_dataset.prepare(slc)

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        only_label = kwargs.pop("only_label", False)
        data = super(TSDatasetH, self)._prepare_seg(ext_slice, **kwargs)

        ############################## Add market information ###########################
        # If we only need label for testing, we do not need to add market information
        if not only_label:
            marketData = self.get_market_information(ext_slice)
            cols = pd.MultiIndex.from_tuples([("feature", feature) for feature in marketData.columns])
            marketData = pd.DataFrame(marketData.values, columns=cols, index=marketData.index)
            # data = data.iloc[:, :-1].join(marketData).join(data.iloc[:, -1])
            data = pd.merge(data.reset_index().set_index('datetime').iloc[:,:-1], marketData.droplevel('instrument'), left_on='datetime', right_index=True, how='left').reset_index().set_index(['datetime', 'instrument']).join(data.iloc[:,-1])
        #################################################################################
        flt_kwargs = copy.deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = TSDataSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
            fillna_type="ffill+bfill"
        )
        return tsds


class TSRankICDatasetH(DatasetH):
    """
    包含 RankIC、RankICIR 的 TSDatasetH
    (T)ime-(S)eries Dataset (H)andler


    Convert the tabular data to Time-Series data

    Requirements analysis

    The typical workflow of a user to get time-series data for an sample
    - process features
    - slice proper data from data handler:  dimension of sample <feature, >
    - Build relation of samples by <time, instrument> index
        - Be able to sample times series of data <timestep, feature>
        - It will be better if the interface is like "torch.utils.data.Dataset"
    - User could build customized batch based on the data
        - The dimension of a batch of data <batch_idx, feature, timestep>
    """

    DEFAULT_STEP_LEN = 30
    # DEFAULT_STEP_LEN = 1000

    def __init__(self, step_len=DEFAULT_STEP_LEN, label_shift=240, **kwargs):
        self.step_len = step_len
        self.label_shift = label_shift
        super().__init__(**kwargs)

    def config(self, **kwargs):
        if "step_len" in kwargs:
            self.step_len = kwargs.pop("step_len")
        super().config(**kwargs)

    def setup_data(self, **kwargs):
        super().setup_data(**kwargs)
        # make sure the calendar is updated to latest when loading data from new config
        cal = self.handler.fetch(col_set=self.handler.CS_RAW).index.get_level_values("datetime").unique()
        self.cal = sorted(cal)

    @staticmethod
    def _extend_slice(slc: slice, cal: list, step_len: int) -> slice:
        # Dataset decide how to slice data(Get more data for timeseries).
        start, end = slc.start, slc.stop
        start_idx = bisect.bisect_left(cal, pd.Timestamp(start))
        pad_start_idx = max(0, start_idx - step_len)
        pad_start = cal[pad_start_idx]
        return slice(pad_start, end)

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        """
        split the _prepare_raw_seg is to leave a hook for data preprocessing before creating processing data
        NOTE: TSDatasetH only support slc segment on datetime !!!
        """
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super()._prepare_seg(ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        # 构造各因子滚动窗口 RankIC、RankICIR
        ##############################################################################################################
        # 取出特征列和收益列
        feature_cols = [col for col in data.columns if col[0] == "feature"]
        ret_col = ("feature", f"ret_{self.label_shift}")

        # 目标因子 = 所有非 ret_ 开头的特征列
        target_factors = [col for col in feature_cols if not col[1].startswith("ret_")]

        # 存储每个因子的历史截面 RankIC 时间序列
        rankic_df = pd.DataFrame(index=data.index.get_level_values("datetime").unique(),
                                 columns=[f[1] for f in target_factors])

        # 原 RankIC 按时间截面计算（不变）
        for dt, df_dt in data.groupby(level="datetime"):
            y = df_dt[ret_col]
            for factor in target_factors:
                x = df_dt[factor]
                corr = spearmanr(x, y)[0]
                rankic_df.at[dt, factor[1]] = corr

        # 向后移动 N 个周期，避免泄露未来收益
        rankic_df = rankic_df.shift(self.label_shift)
        # 再进行缺失值填充
        rankic_df = rankic_df.fillna(method="ffill")
        rankic_df = rankic_df.astype(np.float32)

        # 2. 滚动计算 RankICIR，窗口大小
        # 这里注意频率，当前计算 Alpha158 因子已经 x 24，所以这里的窗口大小对应的是日频
        # rankic_windows = [60, 120, 240]
        rankic_windows = [240, 720, 1440]
        # rankic_windows = [1, 10, 24]
        # rankic_windows = [720]

        # 创建列名
        for factor in target_factors:
            f_name = factor[1]
            for win in rankic_windows:
                data[("feature", f"{f_name}_RankIC{win}")] = data.index.get_level_values("datetime").map(
                    rankic_df[f_name].rolling(win).mean())
                # data[("feature", f"{f_name}_RankICIR{win}")] = data.index.get_level_values("datetime").map(
                #     rankic_df[f_name].rolling(win).mean() / rankic_df[f_name].rolling(win).std())

        # 3. 删除 ret_ 开头的因子
        ret_cols = [col for col in feature_cols if col[1].startswith("ret_")]
        data.drop(columns=ret_cols, inplace=True)
        # 这里我选择直接删除最前面为 NaN 的数据
        # 构造完所有 RankICIR 后统一 dropna
        # data.dropna(
        #     subset=[('feature', f"{f[1]}_RankICIR{win}") for f in target_factors for win in rankic_windows],
        #     inplace=True)
        data.dropna(subset=[('feature', f"{f[1]}_RankIC{win}") for f in target_factors for win in rankic_windows], inplace=True)
        feature_cols = [col for col in data.columns if col[0] == "feature"]
        label_cols = [col for col in data.columns if col[0] == 'label']
        new_col_order = feature_cols + label_cols
        data = data[new_col_order]
        ##############################################################################################################

        tsds = TSDataSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
        )
        return tsds