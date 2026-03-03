import polars as pl
import numpy as np
from abc import ABC

from typing import(
    Callable,
    Any,
    List,
    Dict,
    Union,
    Literal,
    Tuple,
)

from scipy.stats import (
    t,
    chi2,
    kstwo,
    ttest_ind,
    chi2_contingency,
    ks_2samp,
    _stats_py,
)

from math import sqrt, gcd

Alternative = Literal["two-sided", "less", "greater"]
Method = Literal["auto", "exact", "asymp"]
NanPolicy = Literal["propagate", "omit", "raise"]


class StaticTest(ABC):
    def __init__(self,
                 data: Union[pl.DataFrame, pl.LazyFrame],
                 target_columns: Union[str, List[str]], 
                 label_column: str, 
                 groups_num: int, 
                 k_splits: int, 
                 reliability=0.05, 
                 control_label: int = 0,
                 realization: int = 1):
        self.data = data.drop_nulls() # все null выбрасываются
        self.target_columns = target_columns
        self.label_column = label_column
        self.reliability = reliability
        self.groups_num = groups_num
        self.k_splits = k_splits
        self.control_label = control_label
        self.realization = realization

    def _scipy_calc(self,
                    control_label: int, 
                    test_label: int, 
                    target_column: str, 
                    label_column: str,
                    test_function: Callable,
                    **kwargs):
        control_column = (
                            self.data
                            .filter(pl.col(label_column) == control_label)
                            .select(target_column)
                            .fill_null(np.nan)
                            .collect()
                            .to_numpy()
                            .flatten()
                        )
        test_column =  (
                            self.data
                            .filter(pl.col(label_column) == test_label)
                            .select(target_column)
                            .fill_null(np.nan)
                            .collect()
                            .to_numpy()
                            .flatten()
                        )
    
        result = test_function(control_column, test_column, **kwargs,)
        return {
                    "p-value": result[1],
                    "statistic": result[0],
                    "pass": result[1] < self.reliability,
                }

class Ttest(StaticTest):

    def __init__(self, 
                 data, 
                 target_columns, 
                 label_column, 
                 groups_num, 
                 k_splits, 
                 reliability=0.05, 
                 control_label = 0, 
                 realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)
    
    def calculate(self) -> Dict[str, float]:
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Incorrect target col format! {type(self.target_columns).__name__}")
        
        group_pairs = [(self.control_label, group) for group in range(self.groups_num) if group != self.control_label]
        grouped_df = self.data.group_by(self.label_column)

        pivot_df = (
                    pl.concat([self._aggregator_func(grouped_df, column) for column in self.target_columns])
                    # .with_columns(split=pl.lit(self.k_splits))
                )
        if isinstance(self.data, pl.LazyFrame):
            pivot_df = pivot_df.cache()
        # for split_idx in range(self.k_splits):
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for test, control in group_pairs:
                    if self.realization == 0:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._scipy_calc(control_label=control,
                                                                                                 test_label=test,
                                                                                                 target_column=column,
                                                                                                 label_column=self.label_column,
                                                                                                 test_function=ttest_ind,
                                                                                                 nan_policy='omit')
                    elif self.realization == 1:
                        _, count, mean, std, _= (
                            pivot_df
                            .filter((pl.col("column") == column) & ((pl.col("group") == test) | (pl.col("group") == control)))
                            .collect()
                        )
                        count, mean, std = count.to_numpy(), mean.to_numpy(), std.to_numpy()
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._single_calc(std,
                                                                                                  mean,
                                                                                                 count)
                    else:
                        Warning(f"Realization {self.realization} doesn't exist!")
            result[column] = tmp_dict
        return result
    
    @staticmethod
    def _aggregator_func(g_df, column: str) -> pl.LazyFrame:
        return (
            g_df
            .agg([
                pl.col(column).count().alias("count"),
                pl.col(column).mean().alias("mean"),
                pl.col(column).var().alias("std")
            ])
            .with_columns(
                            column=pl.lit(column)
                        )
        )
    
    def _single_calc(self, current_variance: tuple, current_mean: tuple, currecnt_size: tuple) -> Dict[str, Union[float, str]]:
        similar_var = (current_variance[0] / current_variance[1] < 2 and current_variance[0] / current_variance[1] > 0.5)

        t_stat = self._t_statistics(n_list=currecnt_size,
                                    s_list=current_variance,
                                    mean_list=current_mean,
                                    similar_var=similar_var)

        de_fr = self._degree_fredom(n_list=currecnt_size,
                                    s_list=current_variance,
                                    similar_var=similar_var)

        p_value = t.sf(abs(t_stat), de_fr) * 2

        return {
                    "p-value": p_value,
                    "statistic": abs(t_stat),
                    "pass": p_value < self.reliability,
                }
    
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
        


class ChiSquare(StaticTest):

    def __init__(self, 
                 data, 
                 target_columns, 
                 label_column, 
                 groups_num, 
                 k_splits, 
                 reliability=0.05, 
                 control_label = 0, 
                 realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)

    def calculate(self) -> Dict[str, float]:
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Incorrect target col format! {type(self.target_columns).__name__}")
        
        group_pairs = [(self.control_label, group) for group in range(self.groups_num) if group != self.control_label]
        grouped_df = self.data.group_by(self.label_column)

        pivot_df = (
                    pl.concat([self._aggregator_func(grouped_df, column) for column in self.target_columns])
                    # .with_columns(split=pl.lit(self.k_splits))
                )
        if isinstance(self.data, pl.LazyFrame):
            pivot_df = pivot_df.cache()
        # for split_idx in range(self.k_splits):
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for test, control in group_pairs:
                    if self.realization < 2:
                        if self.realization == 0:
                            Warning("No nessesity to make the scipy realization")

                        test_list = (
                            pivot_df
                            .filter((pl.col("column") == column) & (pl.col("group") == test))
                            .select("freq", "category")
                            .collect()
                            .to_dicts()
                        )

                        control_list = (
                            pivot_df
                            .filter((pl.col("column") == column) & (pl.col("group") == control))
                            .select("freq", "category")
                            .collect()
                            .to_dicts()
                        )

                        # test_dict = {k : v for d in test_list for k, v in d.items()}
                        # control_dict = {k : v for d in control_list for k, v in d.items()}
                        test_dict = {d["category"]: d["freq"] for d in test_list}
                        control_dict = {d["category"]: d["freq"] for d in control_list}

                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._single_calc(test_dict,
                                                                                                  control_dict,
                                                                                                  self.reliability)
                    else:
                        Warning(f"Realization {self.realization} doesn't exist!")
                        
            result[column] = tmp_dict
        return result
    
    @staticmethod
    def _aggregator_func(g_df, column: str) -> pl.LazyFrame:
        return (
            g_df
            .agg([
                pl.col(column).value_counts().alias("_freq")
            ])
            .explode("_freq")
            .with_columns(
                            pl.col('_freq').struct.field("count").cast(pl.Int64).alias("freq"),
                            pl.col('_freq').struct.field(column).alias("category"),
                            pl.lit(column).alias("column")
                        )
            .drop('_freq')
        )
    
    @staticmethod
    def _single_calc(test_dict: Dict[str, Any], control_dict: Dict[str, Any], reliability: float) -> Dict[str, Union[float, bool]]:
        fianal_list = [[], []]
        control_categories = control_dict
        test_categories = list(test_dict.keys())
        all_categories = set(control_categories).union(test_categories)
        
        for category in all_categories:
            if category in control_categories:
                fianal_list[0].append(control_dict[category])
            else:
                fianal_list[0].append(0)

            if category in test_categories:
                fianal_list[1].append(test_dict[category])
            else:
                fianal_list[1].append(0)
        
        result =  chi2_contingency(fianal_list) 
        return {
            "p-value" : result[1],
            "statistic" : result[0],
            "pass" : result[1] < reliability
        }
    
class KStest(StaticTest):

    def __init__(self, data, target_columns, label_column, groups_num, k_splits, reliability=0.05, control_label = 0, realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)

    def calculate(self):
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Incorrect target col format! {type(self.target_columns).__name__}")
        
        group_pairs = [(group, self.control_label) for group in range(self.groups_num) if group != self.control_label]
        # pivot_table = self._statistic_calc()

        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                if self.realization == 0:
                    for test, control in group_pairs:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._scipy_calc(control_label=control,
                                                                                                     test_label=test,
                                                                                                     target_column=column,
                                                                                                     label_column=self.label_column,
                                                                                                     test_function=ks_2samp,)
                    result[column] = tmp_dict
                elif self.realization == 1:
                    result[column] = self._single_calc(column, group_pairs)
                    Warning(f"Realization {self.realization} doesn't exist!")

        return result
    
    def _statistic_calc(self) -> pl.LazyFrame:
        return pl.concat([self._single_calc(column, control_label=self.control_label) for column in self.target_columns]).cache()

    def _single_calc(self,
                    target_column: str, 
                    group_pairs: List[Tuple[int]],
                    binary: bool = True,) -> Dict[str, Union[float, bool]]:
        tmp_table = (
            self.data
            # .drop_nulls()
            .select(target_column, self.label_column)
            .sort([target_column, self.label_column])
            .with_columns(
                groups_count=pl.col(self.label_column).count().over(self.label_column),
                edf_over_group=(
                                    pl.cum_count(target_column)
                                    .over(self.label_column)
                                ),
                        )
            .with_columns(
                control_over_control=(
                                        pl.when(pl.col(self.label_column) == self.control_label)
                                        .then(pl.col("edf_over_group"))
                                        .otherwise(0)
                                        .cum_max()
                                    ),
                _test_over_test=(
                                    pl.when(pl.col(self.label_column) != self.control_label)
                                    .then(pl.col("edf_over_group"))
                                    .otherwise(0)
                                ),
                _test_count=(
                                pl.when(pl.col(self.label_column) != self.control_label)
                                .then(pl.col("groups_count"))
                                .otherwise(0)
                ),
                control_count=(
                    pl.when(pl.col(self.label_column) == self.control_label)
                    .then(pl.col("groups_count"))
                    .otherwise(0)
                    .max()
                )
            )
        )
        tmp_dict = {}
        for control, test in group_pairs:
            table = (
                        tmp_table
                        .filter((pl.col(self.label_column) == control) | (pl.col(self.label_column) == test))
                        .with_columns(
                            test_over_test=pl.col("_test_over_test").cum_max(),
                            test_count=pl.col("_test_count").max()
                        )
                        .with_columns(
                            statistic=(
                                (
                                    pl.col("control_over_control").truediv(pl.col("groups_count")) - 
                                    pl.col("test_over_test").truediv(pl.col("test_count"))
                                ).abs()
                            )
                        )
                        .select(
                            pl.col('statistic').max(),
                            pl.col('control_count').max(),
                            pl.col('groups_count').max()
                        )
                        .with_columns(
                            category=pl.lit(target_column),
                            group=pl.lit(test),
                            )
                    ).collect()
            d = table['statistic'][0]
            n1 = table['control_count'][0]
            n2 = table['groups_count'][0]
            tmp_dict[f"split: {self.k_splits - 1} groups: {control}, {test}"] = self._asymptotic_ks_pvalue(d, n1, n2, self.reliability)
        return tmp_dict
    
    @staticmethod
    def _asymptotic_ks_pvalue(d, n1, n2, reliability, method: Method = "auto", alternative: Alternative = "two-sided"):
        n1, n2 = int(n1), int(n2)
        
        # print(n1, n2)
        mode = method
        MAX_AUTO_N = 10000
        
        g = gcd(n1, n2)
        n1g = n1 // g
        n2g = n2 // g
        
        prob = -np.inf
        
        if mode == "auto":
            mode = "exact" if max(n1, n2) <= MAX_AUTO_N else "asymp"
        elif mode == "exact":
            # If lcm(n1,n2) too large, SciPy switches to asymp.
            # SciPy checks against int32 max via n1g >= iinfo(int32).max / n2g.
            if n1g >= np.iinfo(np.int32).max / n2g:
                mode = "asymp"

        if mode == "exact":
            # This is the same internal helper SciPy uses; it may change across SciPy versions.
            success, d2, prob2 = _stats_py._attempt_exact_2kssamp(n1, n2, g, d, alternative)
            if success:
                pvalue = float(np.clip(prob2, 0.0, 1.0))
                return {
                            "value" : pvalue,
                            "statistic" : d,
                            "pass" :  pvalue < reliability
                        }
            # fallback
            mode = "asymp"
        

        # asymptotic
        # Ensure float to avoid overflow; sorted because one-sided formula not symmetric
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == "two-sided":
            # matches SciPy: kstwo.sf(d, round(en))
            prob = kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            expt = -2.0 * z**2 - 2.0 * z * (m + 2.0 * n) / np.sqrt(m * n * (m + n)) / 3.0
            prob = np.exp(expt)
        pvalue = float(np.clip(prob, 0.0, 1.0))

        return {
                    "value" : pvalue,
                    "statistic" : d,
                    "pass" :  pvalue < reliability
                }