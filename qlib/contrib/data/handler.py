# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
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
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
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
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha191(DataHandlerLP):
    def __init__(
        self,
        benchmark="SH000300",
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        custom_feature_tuple=None,
        **kwargs,
    ):
        self.benchmark = benchmark
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        # feature_formula, feature_name = self.get_feature_config()

        feature_list = ([], [])
        feature_list = self.merge_tuples(feature_list, self.get_feature_config())
        if custom_feature_tuple:
            feature_list = self.merge_tuples(feature_list, custom_feature_tuple)

        data_loader = kwargs.get(
            "data_loader",
            {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": feature_list,
                        "label": kwargs.get("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    # "inst_processors": kwargs.get("inst_processors", None),
                },
            },
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def merge_tuples(self, tuple1, tuple2):
        merged_first = tuple1[0] + tuple2[0]  # 合并第一个列表
        merged_second = tuple1[1] + tuple2[1]  # 合并第二个列表
        return (merged_first, merged_second)

    def get_feature_config(self):
        return self.parse_config_to_fields()

    @staticmethod
    def get_label_config():
        return ["Ref($close, -10)/$close - 1"], ["LABEL0"]

    def parse_config_to_fields(self):
        """create factors from config"""
        f_return = "($close/Ref($close, 1)-1)"
        f_hd = "$high-Ref($high, 1)"
        f_ld = "Ref($low, 1)-$low"
        f_dtm = "If($open<=Ref($open, 1), 0, Greater($high-$open, $open-Ref($open, 1)))"
        f_dbm = "If($open>=Ref($open, 1), 0, Greater($open-$low, $open-Ref($open, 1)))"
        f_tr = "Greater(Greater($high-$low, Abs($high-Ref($close, 1))), Abs($low-Ref($close, 1)))"

        alpha_components = {
            "alpha191_001": "-1*Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close-$open)/$open), 6)",
            "alpha191_002": "-1 * Delta((2*$close-$high-$low)/($high+$low+1e-10), 1)",
            "alpha191_003": f"Sum(If($close==Ref($close,1), 0, $close-If($close>Ref($close,1), Less($low,Ref($close,1)), Greater($high,Ref($close,1)))),6)",
            "alpha191_004": f"If(Sum($close, 8) / 8 + Std($close, 8) < Sum($close, 2) / 2, -1, If(Sum($close, 2) / 2 < Sum($close, 8) / 8 - Std($close, 8), 1, If(Or((1 < ($volume / Mean($volume,20))), (($volume / Mean($volume,20)) == 1)), 1, (-1 * 1))))",

            # "alpha191_005": f"(-1 * Max(Corr(Rank($volume, 5), Rank($high, 5), 5), 3))",
            "alpha191_006": f"(CSRank(Sign(Delta(((($open * 0.85) + ($high * 0.15))), 4)))* -1)",


            "alpha191_009": f"EMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/$volume, 2/7)",  # TODO check sma/ema
            "alpha191_010": f"CSRank(Greater(Power(If({f_return} < 0, Std({f_return}, 20), $close), 2),5))",
            "alpha191_011": f"Sum((($close-$low)-($high-$close))/($high-$low)*$volume,6)",


            "alpha191_014": f"$close-Ref($close,5)",
            "alpha191_015": f"$open/Ref($close,1)-1",


            "alpha191_018": f"$close/Ref($close,5)",
            "alpha191_019": f"If($close<Ref($close,5), ($close-Ref($close,5))/Ref($close,5), If($close==Ref($close,5), 0, ($close-Ref($close,5))/$close))",
            "alpha191_020": f"($close-Ref($close,6))/Ref($close,6)*100",
            "alpha191_021": f"Slope(Mean($close,6), 6)",
            "alpha191_022": f"EMA((($close-Mean($close,6))/Mean($close,6)-Ref(($close-Mean($close,6))/Mean($close,6),3)),1/12)",
            "alpha191_023": f"EMA(If($close>Ref($close,1), Std($close,20), 0), 1/20)/(EMA(If($close>Ref($close,1), Std($close,20), 0),1/20)+EMA(If($close<=Ref($close,1),Std($close,20),0),1/20))*100",
            "alpha191_024": f"EMA($close-Ref($close,5), 1/5)",

            "alpha191_025": f"((-1 * CSRank((Delta($close, 7) * (1 - CSRank(WMA(($volume / Mean($volume,20)), 9)))))) * (1 +CSRank(Sum({f_return}, 250))))",

            "alpha191_027": f"WMA(($close-Ref($close,3))/Ref($close,3)*100+($close-Ref($close,6))/Ref($close,6)*100,12)",
            "alpha191_028": f"3*EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9))*100,1/3)-2*EMA(EMA(($close-Min($low,9))/(Greater($high,9)-Max($low,9))*100,1/3),1/3)",
            "alpha191_029": f"($close-Ref($close,6))/Ref($close,6)*$volume",
            # "alpha191_030": f"WMA((REGRESI($close/Ref($close)-1,MKT,SMB,HML, 60))^2,20)", ## cannot calculate multi variate reg now
            "alpha191_031": f"($close-Mean($close,12))/Mean($close,12)*100",
            "alpha191_032": f"(-1 * Sum(CSRank(Corr(CSRank($high), CSRank($volume), 3)), 3))",
            "alpha191_033": f"((((-1 * Min($low, 5)) + Ref(Min($low, 5), 5)) * CSRank(((Sum({f_return}, 240) - Sum({f_return}, 20)) / 220))) *Rank($volume, 5))",
            "alpha191_034": f"Mean($close,12)/$close",
            "alpha191_035": f"(Less(CSRank(WMA(Delta($open, 1), 15)), CSRank(WMA(Corr(($volume), (($open * 0.65) +($open *0.35)), 17),7))) * -1)",
            #
            "alpha191_037": f"(-1 * CSRank(((Sum($open, 5) * Sum({f_return}, 5)) - Ref((Sum($open, 5) * Sum({f_return}, 5)), 10))))",
            "alpha191_038": f"If((Sum($high,20)/20)<$high,-1*Delta($high,2),0)",
            "alpha191_040": f"Sum(If($close>Ref($close,1),$volume,0),26)/Sum(If($close<=Ref($close,1),$volume,0),26)*100",

            "alpha191_042": f"((-1 * CSRank(Std($high, 10))) * Corr($high, $volume, 10))",
            "alpha191_043": f"Sum(If($close>Ref($close,1),$volume,If($close<Ref($close,1),-1 * $volume,0)),6)",

            "alpha191_046": f"(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/(4*$close)",
            "alpha191_047": f"EMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6))*100,1/9)",
            "alpha191_048": f"(-1*((CSRank(((Sign(($close - Ref($close, 1))) + Sign((Ref($close, 1)  - Ref($close, 2)))) +Sign((Ref($close, 2) - Ref($close, 3)))))) * Sum($volume, 5)) / Sum($volume, 20))",
            "alpha191_049": f"Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))",
            "alpha191_050": f"Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))-Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If($high+$low>=Ref($high,1)+Ref($low,1), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))",
            "alpha191_051": f"Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)/(Sum(If(($high+$low)<=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)),Abs($low-Ref($low,1)))),12)+Sum(If(($high+$low)>=(Ref($high,1)+Ref($low,1)), 0, Greater(Abs($high-Ref($high,1)), Abs($low-Ref($low,1)))),12))",
            "alpha191_052": f"Sum(Greater(0,$high-Ref(($high+$low+$close)/3,1)),26)/Sum(Greater(0,Ref(($high+$low+$close)/3,1)-$low),26)*100",
            "alpha191_053": f"Sum($close>Ref($close,1),12)/12*100",
            "alpha191_054": f"(-1 * CSRank((Std(Abs($close - $open), 252) + ($close - $open)) + Corr($close, $open,10)))",
            "alpha191_055": f"Sum(16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/If(And(Abs($high-Ref($close,1))>Abs($low-Ref($close,1)), Abs($high-Ref($close,1))>Abs($high-Ref($low,1))), Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, If(And(Abs($low-Ref($close,1))>Abs($high-Ref($low,1)), Abs($low-Ref($close,1))>Abs($high-Ref($close,1))), Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4))*Greater(Abs($high-Ref($close,1)),Abs($low-Ref($close,1))),20)",
            "alpha191_056": f"(CSRank(($open-Min($open,12)))<CSRank(Power(CSRank(Corr(Sum((($high+$low)/2),19),Sum(Mean($volume,40), 19), 13)), 5)))",
            "alpha191_057": f"EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9))*100,1/3)",
            "alpha191_058": f"Sum($close>Ref($close,1),20)/20*100",  # count if
            "alpha191_059": f"Sum(If($close==Ref($close,1), 0, $close-If($close>Ref($close,1), Less($low,Ref($close,1)), Greater($high,Ref($close,1)))),20)",
            "alpha191_060": f"Sum((($close-$low)-($high-$close))/($high-$low)*$volume,20)",
            ############
            "alpha191_062": f"(-1 * Corr($high, CSRank($volume), 5))",
            "alpha191_063": f"EMA(Greater($close-Ref($close,1),0),1/6)/EMA(Abs($close-Ref($close,1)),1/6)*100",

            "alpha191_065": f"Mean($close,6)/$close",
            "alpha191_066": f"($close-Mean($close,6))/Mean($close,6)*100",
            "alpha191_067": f"EMA(Greater($close-Ref($close,1),0),1/24)/EMA(Abs($close-Ref($close,1)),1/24)*100",
            "alpha191_068": f"EMA((($high+$low)/2-(Ref($high,1)+Ref($low,1))/2)*($high-$low)/$volume,2/15)",
            "alpha191_069": f"If(Sum({f_dtm},20)>Sum({f_dbm},20), (Sum({f_dtm},20)-Sum({f_dbm},20))/Sum({f_dtm},20), If(Sum({f_dtm},20)==Sum({f_dbm},20), 0, (Sum({f_dtm},20)-Sum({f_dbm},20))/Sum({f_dbm},20)))",
            # "alpha191_070": f"Std($money,6)",
            "alpha191_071": f"($close-Mean($close,24))/Mean($close,24)*100",
            "alpha191_072": f"EMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6))*100,1/15)",


            "alpha191_075": f"Sum(And($close>$open, ChangeInstrument('{self.benchmark}', $close)<ChangeInstrument('{self.benchmark}', $open)), 50)/Sum(ChangeInstrument('{self.benchmark}', $close)<ChangeInstrument('{self.benchmark}', $open) ,50)",  # count if
            "alpha191_076": f"Std(Abs(($close/Ref($close,1)-1))/$volume,20)/Mean(Abs(($close/Ref($close,1)-1))/$volume,20)",

            "alpha191_078": f"(($high+$low+$close)/3-Mean(($high+$low+$close)/3,12))/(0.015*Mean(Abs($close-Mean(($high+$low+$close)/3,12)),12))",
            "alpha191_079": f"EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100",
            "alpha191_080": f"($volume-Ref($volume,5))/Ref($volume,5)*100",
            # "alpha191_081": f"EMA($volume,2/21)", # 原始表达式 ！！！！！！！！！！！！！！！！！！
            "alpha191_081": f"Log(EMA($volume,2/21))",
            "alpha191_082": f"EMA((Max($high,6)-$close)/(Max($high,6)-Min($low,6))*100,1/20)",
            # "alpha191_083": f"(-1 * CSRank(Cov(CSRank($high), CSRank($volume), 5)))", # 原始表达式 ！！！！！！！！！！！！！！！！！！
            "alpha191_083": f"Corr(CSRank($high), CSRank($volume), 5)",
            "alpha191_084": f"Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -1 * $volume, 0)),20)",
            "alpha191_085": f"(Rank(($volume / Mean($volume,20)), 20) * Rank((-1 * Delta($close, 7)), 8))",
            "alpha191_086": f"If((0.25 < (((Ref($close, 20) - Ref($close, 10)) / 10) - ((Ref($close, 10) - $close) / 10))), (-1 * 1), If(((((Ref($close, 20) - Ref($close, 10)) / 10) - ((Ref($close, 10) - $close) / 10)) < 0), 1, -1 * ($close - Ref($close, 1))))",

            "alpha191_088": f"($close-Ref($close,20))/Ref($close,20)*100",
            "alpha191_089": f"2*(EMA($close,2/13)-EMA($close,2/27)-EMA(EMA($close,2/13)-EMA($close,2/27),2/10))",

            "alpha191_091": f"((CSRank(($close - Greater($close, 5)))*CSRank(Corr((Mean($volume,40)), $low, 5))) * -1)",

            "alpha191_093": f"Sum(If($open>=Ref($open,1), 0, Greater(($open-$low),($open-Ref($open,1)))),20)",
            "alpha191_094": f"Sum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -1 * $volume, 0)),30)",
            # "alpha191_095": f"Std($money,20)",
            "alpha191_096": f"EMA(EMA(($close-Min($low,9))/(Max($high,9)-Min($low,9))*100,1/3),1/3)",
            # "alpha191_097": f"Std($volume,10)",  # 原始表达式 ！！！！！！！！！！！！！！！！！！
            "alpha191_097": f"Log(Std($volume,10))",
            "alpha191_098": f"If(Or(((Delta((Sum($close, 100) / 100), 100) / Ref($close, 100)) < 0.05), ((Delta((Sum($close, 100) / 100), 100) /Ref($close, 100)) == 0.05)), -1 * ($close - Min($close, 100)), -1 * Delta($close, 3))",
            "alpha191_099": f"(-1 * CSRank(Cov(CSRank($close), CSRank($volume), 5)))",
            # "alpha191_100": f"Std($volume,20)",   # 原始表达式 ！！！！！！！！！！！！！！！！！！
            "alpha191_100": f"Log(Std($volume,20))",
            "alpha191_102": f"EMA(Greater($volume-Ref($volume,1),0),1/6)/EMA(Abs($volume-Ref($volume,1)),1/6)*100",
            "alpha191_103": f"(IdxMin($low,20)/20)*100",
            "alpha191_104": f"(-1 * (Delta(Corr($high, $volume, 5), 5) * CSRank(Std($close, 20))))",
            "alpha191_105": f"(-1 * Corr(CSRank($open), CSRank($volume), 10))",
            "alpha191_106": f"$close-Ref($close,20)",
            "alpha191_107": f"(((-1 * CSRank(($open - Ref($high, 1)))) * CSRank(($open - Ref($close, 1)))) * CSRank(($open - Ref($low, 1))))",

            "alpha191_109": f"EMA($high-$low,2/10)/EMA(EMA($high-$low,2/10),2/10)",
            "alpha191_110": f"Sum(Greater(0,$high-Ref($close,1)),20)/Sum(Greater(0,Ref($close,1)-$low),20)*100",
            "alpha191_111": f"EMA($volume*(($close-$low)-($high-$close))/($high-$low),2/11)-EMA($volume*(($close-$low)-($high-$close))/($high-$low),2/4)",
            "alpha191_112": f"(Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)-Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12))/(Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)+Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12))*100",
            "alpha191_113": f"(-1 * ((CSRank((Sum(Ref($close, 5), 20) / 20)) * Corr($close, $volume, 2)) * CSRank(Corr(Sum($close, 5),Sum($close, 20), 2))))",

            "alpha191_115": f"Power(CSRank(Corr((($high * 0.9) + ($close * 0.1)), Mean($volume,30), 10)), CSRank(Corr(Rank((($high + $low) /2), 4), Rank($volume, 10), 7)))",
            "alpha191_116": f"Slope($close, 20)",
            "alpha191_117": f"((Rank($volume, 32) * (1 - Rank((($close + $high) - $low), 16))) * (1 - Rank({f_return}, 32)))",
            "alpha191_118": f"Sum($high-$open,20)/Sum($open-$low,20)*100",



            "alpha191_122": f"(EMA(EMA(EMA(Log($close),2/13),2/13),2/13)-Ref(EMA(EMA(EMA(Log($close),2/13),2/13),2/13),1))/Ref(EMA(EMA(EMA(Log($close),2/13),2/13),2/13),1)",
            "alpha191_123": f"((CSRank(Corr(Sum((($high + $low) / 2), 20), Sum(Mean($volume,60), 20), 9)) < CSRank(Corr($low, $volume,6))) * -1)",


            "alpha191_126": f"($close+$high+$low)/3",
            "alpha191_127": f"Power(Mean(Power(100*($close-Greater($close,12))/(Greater($close,12)),2), 12), 1/2)",
            "alpha191_128": f"100-(100/(1+Sum(If(($high+$low+$close)/3>Ref(($high+$low+$close)/3,1), ($high+$low+$close)/3*$volume, 0),14)/Sum(If(($high+$low+$close)/3<Ref(($high+$low+$close)/3,1), ($high+$low+$close)/3*$volume, 0), 14)))",
            "alpha191_129": f"Sum(If($close-Ref($close,1)<0, Abs($close-Ref($close,1)), 0),12)",


            # "alpha191_132": f"Mean($money,20)",
            "alpha191_133": f"(IdxMax($high,20)/20)*100-(IdxMin($low,20)/20)*100",
            "alpha191_134": f"($close-Ref($close,12))/Ref($close,12)*$volume",
            "alpha191_135": f"EMA(Ref($close/Ref($close,20),1),1/20)",
            "alpha191_136": f"((-1 * CSRank(Delta({f_return}, 3))) * Corr($open, $volume, 10))",
            "alpha191_137": f"16*($close-Ref($close,1)+($close-$open)/2+Ref($close,1)-Ref($open,1))/(If(And(Abs($high-Ref($close,1))>Abs($low-Ref($close,1)), Abs($high-Ref($close,1))>Abs($high-Ref($low,1))), Abs($high-Ref($close,1))+Abs($low-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, If(And(Abs($low-Ref($close,1))>Abs($high-Ref($low,1)), Abs($low-Ref($close,1))>Abs($high-Ref($close,1))), Abs($low-Ref($close,1))+Abs($high-Ref($close,1))/2+Abs(Ref($close,1)-Ref($open,1))/4, Abs($high-Ref($low,1))+Abs(Ref($close,1)-Ref($open,1))/4)))*Greater(Abs($high-Ref($close,1)),Abs($low-Ref($close,1)))",

            "alpha191_139": f"(-1 * Corr($open, $volume, 10))",
            "alpha191_140": f"Less(CSRank(WMA(((CSRank($open)+CSRank($low))-(CSRank($high)+CSRank($close))), 8)),Rank(WMA(Corr(Rank($close, 8), Rank(Mean($volume,60), 20), 8), 7), 3))",
            "alpha191_141": f"(CSRank(Corr(CSRank($high), CSRank(Mean($volume,15)), 9))* -1)",
            "alpha191_142": f"(((-1 * CSRank(Rank($close, 10))) * CSRank(Delta(Delta($close, 1), 1))) * CSRank(Rank(($volume/Mean($volume,20)), 5)))",
            # "alpha191_143": f"If($close>Ref($close,1), ($close-Ref($close,1))/Ref($close,1)*SELF, SELF)",  初值无法确定
            # "alpha191_144": f"Sum(If($close<Ref($close,1), Abs($close/Ref($close,1)-1)/$money, 0),20)/Sum($close<Ref($close,1),20)",  # count if
            "alpha191_145": f"(Mean($volume,9)-Mean($volume,26))/Mean($volume,12)*100",
            "alpha191_146": f"Mean(($close-Ref($close,1))/Ref($close,1)-EMA(($close-Ref($close,1))/Ref($close,1),2/61),20)*(($close-Ref($close,1))/Ref($close,1)-EMA(($close-Ref($close,1))/Ref($close,1),2/61))/EMA(((($close-Ref($close,1))/Ref($close,1)-EMA(($close-Ref($close,1))/Ref($close,1),2/61))), 2 / 61)",
            "alpha191_147": f"Slope(Mean($close,12), 12)",
            "alpha191_148": f"((CSRank(Corr(($open), Sum(Mean($volume,60), 9), 6)) < CSRank(($open - Min($open, 14)))) * -1)",
            # "alpha191_149": f"REGBETA(FILTER($close/Ref($close,1)-1,BANCHMARKINDEX$close<Ref(BANCHMARKINDEX$close,1)),FILTER(BANCHMARKINDEX$close/Ref(BANCHMARKINDEX$close,1)-1,BANCHMARKINDEX$close<DELAY(BANCHMARKINDEX$close,1)),252)", REGBETA
            # "alpha191_150": f"Log(CSRank(($close+$high+$low)/3*$volume))",
            "alpha191_150": f"Log(($close+$high+$low)/3*$volume)",  # 和上面的这个好像是一样的
            # "alpha191_150": f"($close+$high+$low)/3*$volume",  # 原始表达式 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            "alpha191_151": f"EMA($close-Ref($close,20),1/20)",
            "alpha191_152": f"EMA(Mean(Ref(EMA(Ref($close/Ref($close,9),1),1/9),1),12)-Mean(Ref(EMA(Ref($close/Ref($close,9),1),1/9),1),26),1/9)",
            "alpha191_153": f"(Mean($close,3)+Mean($close,6)+Mean($close,12)+Mean($close,24))/4",

            "alpha191_155": f"EMA($volume,2/13)-EMA($volume,2/27)-EMA(EMA($volume,2/13)-EMA($volume,2/27),2/10)",
            ######################################

            "alpha191_157": f"(Less(Prod(CSRank(CSRank(Log(Sum(Min(CSRank(CSRank((-1 * CSRank(Delta(($close - 1), 5))))), 2), 1)))), 1), 5) +Rank(Ref((-1 * {f_return}), 6), 5))",
            "alpha191_158": f"(($high-EMA($close,2/15))-($low-EMA($close,2/15)))/$close",
            "alpha191_159": f"(($close-Sum(Less($low,Ref($close,1)),6))/Sum(Greater($high, Ref($close,1))-Less($low,Ref($close,1)),6)*12*24+($close-Sum(Less($low,Ref($close,1)),12))/Sum(Greater($high,Ref($close,1))-Less($low,Ref($close,1)),12)*6*24+($close-Sum(Less($low,Ref($close,1)),24))/Sum(Greater($high,Ref($close,1))-Less($low,Ref($close,1)),24)*6*24)*100/(6*12+6*24+12*24)",
            "alpha191_160": f"EMA(If($close<=Ref($close,1), Std($close,20), 0),1/20)",
            "alpha191_161": f"Mean(Greater(Greater(($high-$low),Abs(Ref($close,1)-$high)),Abs(Ref($close,1)-$low)),12)",
            "alpha191_162": f"(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100-Less(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100,12))/(Greater(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100,12)-Less(EMA(Greater($close-Ref($close,1),0),1/12)/EMA(Abs($close-Ref($close,1)),1/12)*100,12))",

            "alpha191_164": f"EMA((If(($close>Ref($close,1)),1/($close-Ref($close,1)),1)-Less(If(($close>Ref($close,1)),1/($close-Ref($close,1)),1),12))/($high-$low)*100,2/13)",
            # "alpha191_165": f"Greater(SumAC($close-Mean($close,48)))-Less(SumAC($close-Mean($close,48)))/Std($close,48) unknown function sumac
            "alpha191_166": f"-87.17797887*Sum($close/Ref($close,1)-Mean($close/Ref($close,1),20),20)/(18*Power(Sum(Power(($close/Ref($close,1)-Mean($close/Ref($close,1), 20)),2),20),1.5))",
            "alpha191_167": f"Sum(If($close-Ref($close,1)>0, $close-Ref($close,1), 0),12)",
            "alpha191_168": f"(-1*$volume/Mean($volume,20))",
            "alpha191_169": f"EMA(Mean(Ref(EMA($close-Ref($close,1),1/9),1),12)-Mean(Ref(EMA($close-Ref($close,1),1/9),1),26),1/10)",

            "alpha191_171": f"((-1 * (($low - $close) * Power($open, 5))) / (($close - $high) * Power($close, 5)))",
            "alpha191_172": f"Mean(Abs(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)-Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))/(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)+Sum(If(And({f_hd}>0, {f_hd}>{f_ld}),{f_hd},0),14)*100/Sum({f_tr},14))*100,6)",
            "alpha191_173": f"3*EMA($close,2/13)-2*EMA(EMA($close, 2/13), 2/13)+EMA(EMA(EMA(Log($close),2/13),2/13),2/13)",
            "alpha191_174": f"EMA(If($close>Ref($close,1), Std($close,20), 0),1/20)",
            "alpha191_175": f"Mean(Greater(Greater(($high-$low),Abs(Ref($close,1)-$high)),Abs(Ref($close,1)-$low)),6)",
            "alpha191_176": f"Corr(CSRank((($close - Min($low, 12)) / (Max($high, 12) - Min($low,12)))), CSRank($volume), 6)",
            "alpha191_177": f"(IdxMax($high,20)/20)*100",
            "alpha191_178": f"($close-Ref($close,1))/Ref($close,1)*$volume",

            "alpha191_180": f"If(Mean($volume,20) < $volume, -1 * Rank(Abs(Delta($close, 7)), 60) * Sign(Delta($close, 7)), (-1 *$volume))",
            "alpha191_181": f"Sum((($close/Ref($close,1)-1)-Mean(($close/Ref($close,1)-1),20))-Power(ChangeInstrument('{self.benchmark}', $close)-Mean(ChangeInstrument('{self.benchmark}', $close),20), 2),20)/Sum(Power(ChangeInstrument('{self.benchmark}', $close)-Mean(ChangeInstrument('{self.benchmark}', $close),20), 3),20)",
            "alpha191_182": f"Sum(Or(And($close>$open, ChangeInstrument('{self.benchmark}', $close)>ChangeInstrument('{self.benchmark}', $open)), And($close<$open, ChangeInstrument('{self.benchmark}', $close)<ChangeInstrument('{self.benchmark}', $open))),20)/20",
            # "alpha191_183": f"Greater(SumAC($close-Mean($close,24)))-Less(SumAC($close-Mean($close,24)))/Std($close,24)", unknow sumac
            "alpha191_184": f"(CSRank(Corr(Ref(($open - $close), 1), $close, 200)) + CSRank(($open - $close)))",
            "alpha191_185": f"CSRank(-1 * Power(1 - ($open / $close), 2))",
            "alpha191_186": f"(Mean(Abs(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)-Sum(If(And({f_hd}>0,{f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))/(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)+Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))*100, 6)+Ref(Mean(Abs(Sum(If(And({f_ld}>0,{f_ld}>{f_hd}), {f_ld}, 0), 14)*100/Sum({f_tr},14)-Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))/(Sum(If(And({f_ld}>0, {f_ld}>{f_hd}), {f_ld}, 0),14)*100/Sum({f_tr},14)+Sum(If(And({f_hd}>0, {f_hd}>{f_ld}), {f_hd}, 0),14)*100/Sum({f_tr},14))*100,6),6))/2",
            "alpha191_187": f"Sum(If($open<=Ref($open,1), 0, Greater(($high-$open),($open-Ref($open,1)))),20)",
            "alpha191_188": f"(($high-$low-EMA($high-$low, 2/11))/EMA($high-$low, 2/11))*100",
            "alpha191_189": f"Mean(Abs($close-Mean($close,6)),6)",
            "alpha191_190": f"Log((Sum($close/Ref($close, 1)>Power($close/Ref($close,19), 1/20),20)-1)*Sum(If($close/Ref($close, 1)<Power($close/Ref($close,19), 1/20),  Power($close/Ref($close, 1)-Power($close/Ref($close,19),1/20), 2), 0), 20 ) / (Sum($close/Ref($close, 1)<Power($close/Ref($close,19), 1/20),20) * (Sum(If($close/Ref($close, 1)>Power($close/Ref($close,19), 1/20), Power($close/Ref($close,1)-Power($close/Ref($close,19), 1/20), 2), 0), 20)))+1e-16)",
            "alpha191_191": f"((Corr(Mean($volume,20), $low, 5) + (($high + $low) / 2)) - $close)",
        }

        return list(alpha_components.values()), list(alpha_components.keys())


class AlphaOLHC(DataHandlerLP):
    def __init__(
        self,
        benchmark="market",
        instruments="all",
        start_time=None,
        end_time=None,
        freq="1min",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        custom_feature_tuple=None,
        **kwargs,
    ):
        self.benchmark = benchmark
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        # feature_formula, feature_name = self.get_feature_config()

        feature_list = ([], [])
        feature_list = self.merge_tuples(feature_list, self.get_feature_config())
        if custom_feature_tuple:
            feature_list = self.merge_tuples(feature_list, custom_feature_tuple)

        data_loader = kwargs.get(
            "data_loader",
            {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": feature_list,
                        "label": kwargs.get("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": kwargs.get("inst_processors", None),
                },
            },
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def merge_tuples(self, tuple1, tuple2):
        merged_first = tuple1[0] + tuple2[0]  # 合并第一个列表
        merged_second = tuple1[1] + tuple2[1]  # 合并第二个列表
        return (merged_first, merged_second)

    def get_feature_config(self):
        return self.parse_config_to_fields()

    @staticmethod
    def get_label_config():
        return ["Ref($close, -10)/$close - 1"], ["LABEL0"]

    def parse_config_to_fields(self):
        """create factors from config"""
        alpha_components = {}

        fields = ["close", "open", "high", "low", "volume"]
        offset_now = 0
        offset_prev = 60  # 1 小时前（60 分钟）

        for f in fields:
            # 如果偏移是0，直接用字段名，不用 Ref
            current_val = f"${f}" if offset_now == 0 else f"Ref(${f}, {offset_now})"
            prev_val = f"Ref(${f}, {offset_prev})"
            expr = f"({current_val} - {prev_val}) / {prev_val}"
            name = f"{f}_chg_1h"
            alpha_components[name] = expr

        return list(alpha_components.values()), list(alpha_components.keys())

class AlphaRiskFactors(DataHandlerLP):
    def __init__(
        self,
        benchmark="market",
        instruments="all",
        start_time=None,
        end_time=None,
        freq="60min",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        custom_feature_tuple=None,
        **kwargs,
    ):
        self.benchmark = benchmark
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        # feature_formula, feature_name = self.get_feature_config()

        feature_list = ([], [])
        feature_list = self.merge_tuples(feature_list, self.get_feature_config())
        if custom_feature_tuple:
            feature_list = self.merge_tuples(feature_list, custom_feature_tuple)

        data_loader = kwargs.get(
            "data_loader",
            {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": feature_list,
                        "label": kwargs.get("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": kwargs.get("inst_processors", None),
                },
            },
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def merge_tuples(self, tuple1, tuple2):
        merged_first = tuple1[0] + tuple2[0]  # 合并第一个列表
        merged_second = tuple1[1] + tuple2[1]  # 合并第二个列表
        return (merged_first, merged_second)

    def get_feature_config(self):
        return self.parse_config_to_fields()

    @staticmethod
    def get_label_config():
        return ["Ref($close, -10)/$close - 1"], ["LABEL0"]

    # def parse_config_to_fields(self):
    #     """构造动量因子字段（Crypto 场景，除数收益率 + EMA + 延迟 + 均值）"""
    #     alpha_components = {}
    #     expr = 'Mean(Ref(EMA(($close/Ref($close,1)-1)-Mask($close/Ref($close,1)-1,"MARKET"),719),3),45)'
    #     alpha_components["momentum_barra"] = expr
    #     return list(alpha_components.values()), list(alpha_components.keys())

    # def parse_config_to_fields(self):
    #     """构造 Crypto 场景下的动量与波动率因子（Barra 风格）"""
    #     alpha_components = {}
    #
    #     # 动量因子（跳过最近3小时，EMA平滑，2天均值）
    #     expr_mom = 'Mean(Ref(EMA($close / Ref($close, 1) - 1, 12), 3), 48)'
    #     alpha_components["momentum_barra"] = expr_mom
    #
    #     # 波动率因子（跳过最近3小时，EMA平滑绝对收益，2天均值）
    #     expr_vol = 'Mean(Ref(EMA(Abs($close / Ref($close, 1) - 1), 12), 3), 48)'
    #     alpha_components["volatility_barra"] = expr_vol
    #
    #     return list(alpha_components.values()), list(alpha_components.keys())

    def parse_config_to_fields(self):
        """构造 Crypto 场景下的动量、波动率与其他 Barra 风格风险因子"""
        alpha_components = {}

        # 动量因子（跳过最近3小时，EMA平滑，2天均值）
        expr_mom = 'Mean(Ref(EMA($close / Ref($close, 1) - 1, 12), 3), 48)'
        alpha_components["momentum_barra"] = expr_mom

        # 波动率因子（跳过最近3小时，EMA平滑绝对收益，2天均值）
        expr_vol = 'Mean(Ref(EMA(Abs($close / Ref($close, 1) - 1), 12), 3), 48)'
        alpha_components["volatility_barra"] = expr_vol

        # 其他Barra风格因子
        expressions = ['($high-$low)/$open', 'Min($low, 60)/$close', 'Corr($close, Log($volume+1), 30)',
                       'Corr($close, Log($volume+1), 60)', 'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 30)',
                       'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)']
        names = ['CORR30', 'MIN60', 'CORD60', 'KLEN', 'CORD30', 'CORR60']

        for name, expr in zip(names, expressions):
            alpha_components[name] = expr

        return list(alpha_components.values()), list(alpha_components.keys())


class AlphaBestFactors(DataHandlerLP):
    def __init__(
        self,
        benchmark="MARKET",
        instruments="all",
        start_time=None,
        end_time=None,
        freq="60min",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        custom_feature_tuple=None,
        **kwargs,
    ):
        self.benchmark = benchmark
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        # feature_formula, feature_name = self.get_feature_config()

        feature_list = ([], [])
        feature_list = self.merge_tuples(feature_list, self.get_feature_config())
        if custom_feature_tuple:
            feature_list = self.merge_tuples(feature_list, custom_feature_tuple)

        data_loader = kwargs.get(
            "data_loader",
            {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": feature_list,
                        "label": kwargs.get("label", self.get_label_config()),
                    },
                    "filter_pipe": filter_pipe,
                    "freq": freq,
                    "inst_processors": kwargs.get("inst_processors", None),
                },
            },
        )
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
        )

    def merge_tuples(self, tuple1, tuple2):
        merged_first = tuple1[0] + tuple2[0]  # 合并第一个列表
        merged_second = tuple1[1] + tuple2[1]  # 合并第二个列表
        return (merged_first, merged_second)

    def get_feature_config(self):
        return self.parse_config_to_fields()

    @staticmethod
    def get_label_config():
        return ["Ref($close, -240)/$close - 1"], ["LABEL0"]

    def parse_config_to_fields(self):
        """构造 Crypto 场景下的动量、波动率与其他 Barra 风格风险因子"""
        alpha_components = {}

        alpha_components["Alpha191_150"] = f"Log(($close+$high+$low)/3*$volume)"

        # 其他Barra风格因子
        expressions = ['($high-$low)/$open', 'Corr($close, Log($volume+1), 60)', 'Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)']
        names = ['KLEN', 'CORR60', 'CORD60']

        for name, expr in zip(names, expressions):
            alpha_components[name] = expr

        return list(alpha_components.values()), list(alpha_components.keys())
