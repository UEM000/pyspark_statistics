from __future__ import annotations
from abc import ABC, abstractmethod

import warnings
import pandas as pd

from typing import (
    Callable, 
    Union, 
    List,
    Dict,
)

from scipy.stats import (
    t,
    chi2_contingency,
    ks_2samp,
    ttest_ind,
)

from math import sqrt

class StatTest(ABC):
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_columns: Union[str, List[str]], 
                 label_column: str, 
                 groups_num: int, 
                 k_splits: int, 
                 reliability=0.05, 
                 control_label: int = 0,
                 realization: int = 1): 
        """
        control_label (int): лэйбел контрольной группы, с которой все остальные будут сравниваться;
        realization(int): выбор реализации теста: 
                        0 - scipy реалзация;
                        1 - итеративный подход;
                        2 - параллельный подход, где результат вычисляется сразу для k сплитов.
        """
        self.data = data
        self.target_columns = target_columns
        self.label_column = label_column
        self.reliability = reliability
        self.groups_num = groups_num
        self.k_splits = k_splits
        self.control_label = control_label
        self.realization = realization

    def _iterative_calc(self, 
                        control_label: int, 
                        test_label: int, 
                        target_column: str, 
                        label_column: str,
                        test_function: Callable,
                        **kwargs):
        result = test_function(
            self.data.loc[self.data[label_column] == control_label, target_column],
            self.data.loc[self.data[label_column] == test_label, target_column],
            **kwargs
        )

        return {
                    "p-value": result.pvalue,
                    "statistic": result.statistic,
                    "pass": result.pvalue < self.reliability,
                }
    
    @abstractmethod
    def _single_column_calc(self,
                            pivot_table: pd.DataFrame,
                            target_column: str, 
                            test: int, 
                            control: int, 
                            split_idx: int) -> dict[str, Union[float, bool]]:
        pass


class TestingTTestPandas(StatTest):
    """Т-тест с учетом случаев разной дисперсии. присутствуют все три реализации."""
    def __init__(self, data, target_columns, label_column, groups_num, k_splits, reliability=0.05, control_label = 0, realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)

    def calculate(self) -> Dict[str, Union[float, bool]]:
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Something go wrong! \n {self.target_columns}")
        
        if self.realization > 0:
            stats_pivot_table = pd.concat([self._stats_over_split(self.data, 
                                                                  self.label_column, 
                                                                  self.target_columns, 
                                                                  split_idx,
                                                                  self.realization) 
                                                                  for split_idx in range(self.k_splits)])
    
        group_lsit = list(range(self.groups_num))
        group_lsit.pop(self.control_label)
        
        groups_pairs_list = [(self.control_label, i) for i in group_lsit]
        
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for control, test in groups_pairs_list:
                    if self.realization == 0:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._iterative_calc(control_label=control,
                                                                                                     test_label=test,
                                                                                                     target_column=column,
                                                                                                     label_column=self.label_column,
                                                                                                     test_function=ttest_ind,
                                                                                                     nan_policy='omit')
                    else:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._single_column_calc(pivot_table=stats_pivot_table,
                                                                                                         target_column=column,
                                                                                                         test=test,
                                                                                                         control=control,
                                                                                                         split_idx=split)
            result[column] = tmp_dict
        
        return result
    
    
    def _single_column_calc(self,
                            pivot_table: pd.DataFrame,
                            target_column: str, 
                            test: int, 
                            control: int,
                            split_idx: int):
        test_mean, test_count, test_var = pivot_table.loc[(split_idx, test), (target_column, slice(None))]
        control_mean, control_count, control_var = pivot_table.loc[(split_idx, control), (target_column, slice(None))]

        similar_var = (control_var / test_var < 2 and control_var / test_var > 0.5)

        t_stat = self._t_statistics(n_list=(control_count, test_count),
                                    s_list=(control_var, test_var),
                                    mean_list=(control_mean, test_mean),
                                    similar_var=similar_var)

        de_fr = self._degree_fredom(n_list=(control_count, test_count),
                                    s_list=(control_var, test_var),
                                    similar_var=similar_var)

        p_value = t.sf(abs(t_stat), de_fr) * 2

        return {
                    "p-value": p_value,
                    "statistic": t_stat,
                    "pass": p_value < self.reliability,
                }

    @staticmethod
    def _stats_over_split(df: pd.DataFrame,
                          group_column: str,
                          target_columns: List[str],
                          k_split: int,
                          realization: int):
        stats = (
                    df
                    .loc[:, [group_column] + target_columns]
                    .groupby(df[group_column].map(lambda x: x[k_split]) if realization == 2 else group_column)
                    [target_columns]
                    .agg(["mean", "count", "var"])
                )
    
        stats.index = pd.MultiIndex.from_product([[k_split], stats.index], names=['split', 'group'])
        return stats
    
    @staticmethod
    def _t_statistics(n_list: tuple, 
                      s_list: tuple, 
                      mean_list: tuple, 
                      similar_var: bool = True) -> float:
        if similar_var:
            sp = sqrt(
                (
                    (n_list[0] - 1) * s_list[0] + 
                    (n_list[1] - 1) * s_list[1] 
                ) / ( n_list[0] + n_list[1] - 2)
            )
            t_stat = (mean_list[0] - mean_list[1]) / (sp * sqrt(1 / n_list[0] + 1 / n_list[1]))
        else:
            s_delta =sqrt(s_list[0] / n_list[0] + s_list[1] / n_list[1])
            t_stat = (mean_list[0] - mean_list[1]) / s_delta
        
        return t_stat
    
    @staticmethod
    def _degree_fredom(n_list: tuple, 
                       s_list: tuple = (0, 0), 
                       similar_var: bool = True) -> Union[int, float]:
        if similar_var:
            return n_list[0] + n_list[1] - 2
        else:
            de_fr = ((s_list[0] / n_list[0] + s_list[1] / n_list[1]) ** 2) / (
                          (s_list[0] / n_list[0]) ** 2 / (n_list[0] - 1) + (s_list[1] / n_list[1]) ** 2 / (n_list[1] - 1)
                          )
            return de_fr
    
class TestingChiSquarePandas(StatTest):
    """Хи-квадрат тест на однородность с поправкой Йейтса. Присутствут все три реализации"""

    def __init__(self, data, target_columns, label_column, groups_num, k_splits, reliability=0.05, control_label = 0, realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)

    def calculate(self) -> Dict[str, Union[float, bool]]:
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Something go wrong! \n {self.target_columns}")
        
        if self.realization > 0:
            pivot_table = pd.concat([self._pivot_table(self.data,
                                                       self.label_column,
                                                       self.target_columns,
                                                       split_idx,
                                                       self.realization
                                                      ) for split_idx in range(self.k_splits)])
    
        group_lsit = list(range(self.groups_num))
        group_lsit.pop(self.control_label)
        
        groups_pairs_list = [(self.control_label, i) for i in group_lsit]
        
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for control, test in groups_pairs_list:
                    if self.realization == 0:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._iterative_calc(control_label=control,
                                                                                                    test_label=test,
                                                                                                    target_column=column,
                                                                                                    label_column=self.label_column,)
                    else:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._single_column_calc(pivot_table=pivot_table,
                                                                                                         target_column=column,
                                                                                                         test=test,
                                                                                                         control=control,
                                                                                                         split_idx=split)
            result[column] = tmp_dict
        
        return result
    
    def _iterative_calc(self,
                        control_label: int,
                        test_label: int,
                        target_column: str,
                        label_column: str,
                        **kwargs):
        pivot_table = self._pre_processing(self.data, target_column, label_column)

        contingency_table = pd.DataFrame(
            [
                pivot_table.loc[control_label].values,
                pivot_table.loc[test_label].values
            ],
            index=[control_label, test_label],
            columns=pivot_table.columns
        )
        
        contingency_table = contingency_table.loc[:, contingency_table.sum(axis=0) > 0]

        result = chi2_contingency(contingency_table.values, **kwargs)

        return {
                    "p-value": result.pvalue,
                    "statistic": result.statistic,
                    "pass": result.pvalue < self.reliability,
                }
    
    @staticmethod
    def _pre_processing(data: pd.DataFrame, 
                         target_column: str,
                         label_column: str,
                         ) -> pd.DataFrame:
        pivot_table = (
                            data
                            .groupby(label_column)[target_column]
                            .value_counts()
                            .unstack()
                            .fillna(0)
                        )
        return pivot_table

    @staticmethod
    def _pivot_table(df: pd.DataFrame,
                     group_column: str,
                     target_columns: List[str],
                     k_split: int,
                     realization: int) -> pd.DataFrame:
        stats = (
                    df
                    .loc[:, [group_column] + target_columns]
                    .groupby(df[group_column].map(lambda x: x[k_split]) if realization == 2 else group_column)
                    [target_columns]
                    .apply(lambda x: x.apply(pd.Series.value_counts))
                    .unstack()
                    .dropna(axis=1)
                )

        stats.index = pd.MultiIndex.from_product([[k_split], stats.index], names=['split', 'group'])
        return stats
    
    def _single_column_calc(self,
                            pivot_table: pd.DataFrame,
                            target_column: str,
                            test: int,
                            control: int,
                            split_idx: int) -> dict[str, Union[float, bool]]:

        contingency_table = pd.DataFrame(
            [
                pivot_table.loc[(split_idx, control), target_column].sort_index().values,
                pivot_table.loc[(split_idx, test), target_column].sort_index().values
            ],
            index=[control, test],
            columns=pivot_table.columns.get_level_values(1).unique()
        )
        
        contingency_table = contingency_table.loc[:, contingency_table.sum(axis=0) > 0]

        result = chi2_contingency(contingency_table.values)

        return {
                    "p-value": result.pvalue,
                    "statistic": result.statistic,
                    "pass": result.pvalue < self.reliability,
                }

class TestingKStestPandas(StatTest):
    """КС-тест. Присутствует только scipy реализация"""
    def __init__(self, data, target_columns, label_column, groups_num, k_splits, reliability=0.05, control_label = 0, realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)
    
    def calculate(self) -> Dict[str, Union[float, bool]]:
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Something go wrong! \n {self.target_columns}")
    
        group_lsit = list(range(self.groups_num))
        group_lsit.pop(self.control_label)
        
        groups_pairs_list = [(self.control_label, i) for i in group_lsit]
        
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for control, test in groups_pairs_list:
                    if self.realization == 0:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._iterative_calc(control_label=control,
                                                                                                    test_label=test,
                                                                                                    target_column=column,
                                                                                                    label_column=self.label_column,
                                                                                                    test_function=ks_2samp,)
                    else:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = {
                            "p-value" : None,
                            "statistic" : None,
                            "pass" : None
                        }
            result[column] = tmp_dict
        
        return result

    def _single_column_calc(self, pivot_table, target_column, test, control, split_idx):
        """
        KS-тест не поддерживает параллельную реализацию (realization > 0).
        """
        raise NotImplementedError(
            "KS-test only supports realization=0 (scipy implementation). "
            "Parallel/iterative approaches require raw data, not pivot tables."
        )