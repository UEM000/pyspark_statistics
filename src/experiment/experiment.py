from __future__ import annotations

from .config import test_list, experiment_data
from ..statistic_realization.spark_statistic import TestingTTest, TestingChiSquare, TestingKStest
from ..statistic_realization.pandas_statistic import TestingTTestPandas, TestingKStestPandas, TestingChiSquarePandas
from ..statistic_realization.polars_statistic import Ttest, ChiSquare, KStest
from ..utils.spliter import StandartSpliter, BinarySpliter
from ..utils.reporter import Reporter, MemorySparkReporter
from ..utils.enums import DataRealization

from pyspark.sql import SparkSession, DataFrame

from tqdm import tqdm
from IPython.utils import io
from typing import Callable, Dict, Any, Union

import numpy as np
import pandas as pd
import polars as pl
import sys

class Experiment():

    algo_dict = {
            "naive" : 0,
            "iterative" : 1,
            "parallel" : 2,
            "alternative" : 3
        }

    def __init__(self,
                 data: Union[DataFrame, pd.DataFrame],
                 result_path: str,
                 session: SparkSession,
                 report: bool=False,
                 output: bool=False,
                 output_path: str=None):
        self.data = data
        self.test_list = test_list
        self.session = session
        self.result_path = result_path
        self.report = report
        self.output = output
        self.output_path = output_path
        self.data_realization = self._select_data_realization(data)

        self.constants = experiment_data["constant var"]
        self.variables = experiment_data["variative var"]

        mesh = np.meshgrid(*self.variables.values())
        self.permutations =  np.stack([combination.ravel() for combination in mesh], axis=1)
        



        # self.test = None

    def execute(self):
        for experiment in self.test_list:
            for experiment_name, experiment_settings in experiment.items():
                test_name_list = experiment_settings["test_type"]
                algo_list = experiment_settings["algo_type"]
                for test_name in test_name_list:
                    if test_name == "chisquare":
                        categorical = True
                    else:
                        categorical = False
                    for algo in algo_list:
                        self._single_test_execute(experiment_name=experiment_name,
                                                algo=self.algo_dict[algo],
                                                test_name=test_name,
                                                categorical=categorical)

    def _single_test_execute(self,
                             experiment_name: str,
                             algo: int, 
                             test_name: str,
                             categorical: bool):
        number_of_combinations = len(self.permutations)
        test = self._single_test_inicialization(test_name=test_name)
        
        reporter = Reporter()
        groups_num = self.constants["groups_num"]

        
        for combination in tqdm(range(number_of_combinations), 
                                desc=f"Grid Search Progress for experiment={experiment_name}, test={test_name}, algo={algo}", colour='green', 
                                # file=sys.stdout,
                                # disable=True,
                                ):
            # with io.capture_output() as captured:
                cur_params = dict(zip(list(self.variables.keys()), self.permutations[combination]))
                if cur_params['fractions'] != 1:
                    if self.data_realization == DataRealization.spark:
                        cur_data = self.data.sample(fraction=cur_params['fractions'],
                                                    seed=self.constants["random_state"],
                                                    withReplacement=False)
                    elif self.data_realization == DataRealization.pandas:
                        cur_data = self.data.sample(frac=cur_params['fractions'],
                                                    random_state=self.constants["random_state"],
                                                    replace=False)
                    elif self.data_realization == DataRealization.polars:
                        # Оптимизация: используем встроенный sample вместо gather
                        if isinstance(self.data, pl.LazyFrame):
                            cur_data = self.data.sample(
                                fraction=cur_params['fractions'],
                                seed=self.constants["random_state"],
                                with_replacement=False,
                                shuffle=True
                            )
                        else:
                            cur_data = self.data.sample(
                                fraction=cur_params['fractions'],
                                seed=self.constants["random_state"],
                                with_replacement=False,
                                shuffle=True
                            ).lazy()
                else:
                    cur_data = self.data

                if categorical:
                    target_columns = self.constants["target_category_columns"]
                else:
                    target_columns = self.constants["target_numeric_columns"]

                cur_target_cols = np.random.choice(target_columns, 
                                                    int(cur_params['target_frac'] * len(target_columns)), 
                                                    replace=False).tolist()
                reporter.start()
                if algo == 2:
                    spliter = BinarySpliter(cur_data, 
                                            groups_num=groups_num, 
                                            k_splits=int(cur_params['k_splits']))
                    df = reporter.memory_monitor(spliter.split)
                    
                    tester = test(df,
                                cur_target_cols,
                                "group",
                                groups_num,
                                int(cur_params['k_splits']),
                                realization=2)
                    result = reporter.memory_monitor(tester.calculate)

                    if self.output:
                        self._write_otput(result, test_name, algo)
                else:
                    spliter = StandartSpliter(cur_data, groups_num=groups_num)
                    all_results = {} 
                    for split_idx in range(int(cur_params['k_splits'])):
                        df = reporter.memory_monitor(spliter.split, (split_idx))

                        tester = test(df,
                                    cur_target_cols,
                                    "group",
                                    groups_num,
                                    1,
                                    realization=algo)
                        result = reporter.memory_monitor(tester.calculate)

                        # Из-за особенности реализации, при итеративном алгоритме, результат всегда будет писаться как 'split 0'
                        # Поэтому исправим это для удобства чтения
                        for column, splits_data in result.items():
                            if column not in all_results:
                                all_results[column] = {}
                            for key, value in splits_data.items():
                                # Заменяем split: 0 на split: {split_idx}
                                new_key = key.replace("split: 0", f"split: {split_idx + 1}")
                                all_results[column][new_key] = value

                    if self.output:
                        self._write_otput(all_results, test_name, algo)
                reporter.stop()    

                if self.report:
                    self._write_results(reporter, 
                                        algo, 
                                        groups_num,
                                        test_name,
                                        experiment_name,
                                        cur_params)

    def _single_test_inicialization(self, test_name: str) -> Callable:
        if self.data_realization == DataRealization.spark:
            test_dict = {
                "ttest" : TestingTTest,
                "chisquare" : TestingChiSquare,
                "kstest" : TestingKStest
            }
        elif self.data_realization == DataRealization.pandas:
            test_dict = {
                "ttest" : TestingTTestPandas,
                "chisquare" : TestingChiSquarePandas,
                "kstest" : TestingKStestPandas
            }
        elif self.data_realization == DataRealization.polars:
            test_dict = {
                "ttest" : Ttest,
                "chisquare" : ChiSquare,
                "kstest" : KStest
            }
        else:
            raise TypeError("Incorrect type!")
        return test_dict[test_name]
        

    def _write_results(self, 
                       reporter: Reporter, 
                       algo: int, 
                       groups_num: int,
                       test_name: str,
                       experiment_name: str,
                       cur_params: Dict[str, Any]):
        
        with open(self.result_path + f'spliter_test_{algo}_{groups_num}.txt', "a") as f:
            f.write(f"experiment_name={experiment_name}; ")
            f.write(f"test_label={test_name}; ")
            f.write(f"realization={algo}; ")
            for param, value in cur_params.items():
                f.write(f"{param}={value}; ")
            report = reporter.get_report()
            for element, result in report.items():
                f.write(f"{element} = {result:.5f}; ")
            f.write("\n")

    def _write_otput(self, result: dict, test_name: str, algo: int) -> None:
        with open(self.output_path, "a") as f:
                for column, group_split in result.items():
                    for permutation, test in group_split.items():
                        f.write(f"test_label={test_name}; ")
                        f.write(f"realization={algo}; ")
                        f.write(f"column={column}; ")
                        f.write(f"permutation={permutation}; ")
                        for name, value in test.items():
                            f.write(f"{name}={value}; ")
                        f.write("\n")
                        
    @staticmethod
    def _select_data_realization(data):
        if isinstance(data, DataFrame):
            return DataRealization.spark
        elif isinstance(data, pd.DataFrame):
            return DataRealization.pandas
        elif isinstance(data, pl.DataFrame) or isinstance(data, pl.LazyFrame):
            return DataRealization.polars
        else:
            raise Exception("Incorrect data format!")
    