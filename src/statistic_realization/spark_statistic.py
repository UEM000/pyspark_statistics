from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql.types import DoubleType
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F
from typing import (
    List, 
    Union, 
    Callable, 
    Dict, 
    Literal
)
from math import sqrt, gcd, log, ceil

from scipy.stats import (
    t,
    chi2,
    kstwo,
    ttest_ind,
    chi2_contingency,
    ks_2samp,
    _stats_py,
)

import numpy as np
import pandas as pd

Alternative = Literal["two-sided", "less", "greater"]
Method = Literal["auto", "exact", "asymp"]
NanPolicy = Literal["propagate", "omit", "raise"]


class StatTest():
    def __init__(self, 
                 data: DataFrame, 
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
                        0 - тест с выгрузкой данных (scipy); 
                        1 - тест без выгрузки, подсчтет происходит для 1-го сплита;
                        2 - тест без выгрузки, подсчтет происходит для всех сплитов сразу;
                        3 - альтернативная реализация;
        """
        self.data = data.na.drop() # TODO: Все nan выкидываются!
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
                            .filter(F.col(label_column) == control_label)
                            .select(target_column)
                            .fillna(np.nan)
                            .rdd
                            .flatMap(lambda row: row)
                            .collect()
                        )
        test_column =  (
                            self.data
                            .filter(F.col(label_column) == test_label)
                            .select(target_column)
                            .fillna(np.nan)
                            .rdd
                            .flatMap(lambda row: row)
                            .collect()
                        )
        
        result = test_function(control_column, test_column, **kwargs,)
        return {
                    "p-value": result[1],
                    "statistic": result[0],
                    "pass": result[1] < self.reliability,
                }


class TestingTTest(StatTest):
    """
    Тестируем новый метод разбиения: генерируем сразу все выборки и делаем join к данным.
    Потом за один проход для всех разбиений считаем статистику
    """
    def __init__(self, data, target_columns, label_column, groups_num, k_splits, reliability=0.05, control_label = 0, realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)



    def calculate(self) -> dict:

        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Something go wrong! \n {self.target_columns}")

        group_lsit = list(range(self.groups_num))
        group_lsit.pop(self.control_label)
        
        groups_pairs_list = [(self.control_label, i) for i in group_lsit]

        if self.realization == 1 or self.realization == 2:
            statistic_frame = self._statistic_calculation(data=self.data,
                                                        group_column=self.label_column,
                                                        target_columns=self.target_columns,
                                                        groups_num=self.groups_num,
                                                        k_splits=self.k_splits,
                                                        binary=(self.realization - 1)).collect()[0]
        if self.realization == 3:
            statistic_frame = self._alternative_statistic_calc(data=self.data,
                                                        group_column=self.label_column,
                                                        target_columns=self.target_columns).cache()
        
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for control, test in groups_pairs_list:
                    if self.realization:
                        if self.realization == 1 or self.realization == 2:
                            current_variance = (statistic_frame[f"{column}_{split}_{control}_var"],
                                                statistic_frame[f"{column}_{split}_{test}_var"])
                            
                            current_mean = (statistic_frame[f"{column}_{split}_{control}_mean"],
                                                statistic_frame[f"{column}_{split}_{test}_mean"])
                            
                            currecnt_size = (statistic_frame[f"{column}_{split}_{control}_count"],
                                                statistic_frame[f"{column}_{split}_{test}_count"])
                        elif self.realization == 3:
                            currecnt_size, current_mean, current_variance = self._extract_stats(statistic_frame,
                                                                                                column,
                                                                                                control,
                                                                                                test,
                                                                                                split)

                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._single_calc(current_variance,
                                                                                                    current_mean,
                                                                                                    currecnt_size)
                    else:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._scipy_calc(control_label=control,
                                                                                                 test_label=test,
                                                                                                 target_column=column,
                                                                                                 label_column=self.label_column,
                                                                                                 test_function=ttest_ind,
                                                                                                 nan_policy='omit')
            result[column] = tmp_dict
            
        return result
    
    def _single_calc(self, current_variance: tuple, current_mean: tuple, currecnt_size: tuple) -> dict:
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
    def _statistic_calculation(data: DataFrame, 
                          group_column: str, 
                          target_columns: List[str], 
                          groups_num: int, 
                          k_splits: int,
                          binary: bool = True) -> DataFrame:
        splits = range(k_splits)
        group_indeces = range(groups_num)

        agg_expr = []

        for idx in splits:
            byte_pos = idx + 1
            if binary:
                expression = f"ascii(substring({group_column}, {byte_pos}, 1))"
            else:
                expression = f"{group_column}"
            group_mark = F.expr(expression)

            for column in target_columns:
                for group in group_indeces:
                    condition = (group == group_mark)
                    agg_expr.extend([
                        F.count(F.when(condition, 1)).alias(f"{column}_{idx}_{group}_count"),
                        F.mean(F.when(condition, F.col(column))).alias(f"{column}_{idx}_{group}_mean"),
                        F.variance(F.when(condition, F.col(column))).alias(f"{column}_{idx}_{group}_var")
                    ])

        return data.agg(*agg_expr)
    
    @staticmethod
    def _alternative_statistic_calc(data: DataFrame, 
                                    group_column: str, 
                                    target_columns: List[str]) -> DataFrame:
        agg_expr = []

        for column in target_columns:
                agg_expr.extend([
                        F.count(column).alias(f"{column}_count"),
                        F.mean(column).alias(f"{column}_mean"),
                        F.variance(column).alias(f"{column}_var")
                ])
                

        tmp_table = (
                        data
                        .withColumn("groups_array", 
                                    F.expr(f"transform(split({group_column}, ''), x -> ascii(x))"))
                        .select(
                                *[
                                    F.col(col)    for col in target_columns
                                ],
                                F.posexplode("groups_array")
                        )
                        .withColumnsRenamed({"pos" : "split", "col" : "grouped"})
                        .groupBy(["split", "grouped"])
                        .agg(*agg_expr)
                    )

        return tmp_table
    
    @staticmethod
    def _extract_stats(pivot_table: DataFrame, 
                       target_column:str, 
                       control: int, 
                       test: int, 
                       split: int):
        t_count, t_mean, t_var = (
                                    pivot_table
                                    .filter(((F.col("split") == split) & (F.col("grouped") == ord(str(test)))))
                                    .select(*[
                                            f"{target_column}_count",
                                            f"{target_column}_mean",
                                            f"{target_column}_var"
                                            ])
                                    .collect()[0]
                                )
        
        c_count, c_mean, c_var = (
                                    pivot_table
                                    .filter(((F.col("split") == split) & (F.col("grouped") == ord(str(control)))))
                                    .select(*[
                                            f"{target_column}_count",
                                            f"{target_column}_mean",
                                            f"{target_column}_var"
                                            ])
                                    .collect()[0]
                                )
        return (c_count, t_count), (c_mean, t_mean), (c_var, t_var)
    
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
    
class TestingChiSquare(StatTest):
    """
    Тестируем самописный хи-квадрат тест.
    Выполняется как на одном сплите, так и на всех сплитах сразу.
    """

    def __init__(self, 
                 data, 
                 target_columns, 
                 label_column, 
                 groups_num, # Не используется в этом тесте
                 k_splits, 
                 reliability=0.05, 
                 control_label = 0, 
                 realization = 1):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)

    def calculate(self, correction: bool = True) -> dict:
        """
        correction (bool): Поправка Йейтса, позволяющая улучшить результаты хи-квадарат теста.
                            По умолчанию включена в scipy.
        """
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            raise Exception(f"Something go wrong! \n {self.target_columns}")
        
        # if self.realization == 3:
        #     return self._alternative_calculation()

        if self.realization:
            return self._transform_stats_to_dict(self._spark_calc(self.realization - 1, correction))
        

        group_lsit = list(range(self.groups_num))
        group_lsit.pop(self.control_label)
        
        groups_pairs_list = [(self.control_label, i) for i in group_lsit]

        result = {}

        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for control, test in groups_pairs_list:
                    tmp_dict[f"split: {split} groups: {control}, {test}"] = self._scipy_calc(control_label=control,
                                                                                                 test_label=test,
                                                                                                 target_column=column,
                                                                                                 label_column=self.label_column)
            result[column] = tmp_dict
        
        return result
                
    def _transform_stats_to_dict(self, df: pd.DataFrame) -> Dict[str, Union[float, bool]]:
        result = {}
        for column_name, group_df in df.groupby('column'):
            result[column_name] = {}
            for idx, row in group_df.iterrows():
                key = f"split: {int(row['split'])} group: 0, {int(row['group'])}"
                p_val = float(row['p_value'])
                result[column_name][key] = {
                    'p-value': p_val,
                    'statistic' : float(row['statistic']),
                    'pass' : bool(p_val < self.reliability)
                }    
        return result

    def _spark_calc(self, binary: bool, correction: bool):
        group_column = self.label_column
        # Собираем табличку частот: название колонки | метка | сплит | группа | частота 
        result_table = None
        
        @F.udf(returnType=DoubleType())
        def p_value_calc(statistic: float, dof: int) -> float:
            return float(chi2.sf(statistic, dof))   
        

        for split in range(self.k_splits):
            if binary:
                group_extract_expresion = F.expr(f"ascii(substring({group_column}, {split}, 1))").alias("group")
            else:
                group_extract_expresion = F.col(group_column).alias("group")

            for column in self.target_columns:
                # grand_total_window = Window.partitionBy("split", "column")
                chi_square_window = Window.partitionBy("group")
                category_total_window = Window.partitionBy("category") 
                group_total_window = Window.partitionBy("group")
                tmp_table = (
                    self.data
                        .select(
                            # F.lit(column).alias("column"),
                            F.col(column).alias("category"),
                            # F.lit(split).alias("split"),
                            group_extract_expresion, # номер группы "group"
                            (
                                F.count("*")
                                    .over(Window.partitionBy(column, 
                                                                group_extract_expresion
                                                                ))
                                    .alias("freq")
                            )
                        )
                        # .distinct()
                        .dropDuplicates(["group", "category"])
                        .withColumn("category_total", F.sum("freq").over(category_total_window))
                    )
                
                stats_per_split = (
                                        tmp_table
                                        .withColumn("group_total", F.sum("freq").over(group_total_window))
                                    )
                control_freq = (
                                    stats_per_split        
                                    .withColumn("control_freq",
                                                F.when(F.col("group") == self.control_label, F.col("freq"))
                                                .otherwise(F.lit(0)))
                                    .withColumn("control_group_total",
                                                F.when(F.col("group") == self.control_label, F.col("group_total"))
                                                .otherwise(F.lit(0)))
                                    .withColumn("control_freq_final",
                                                F.max("control_freq").over(category_total_window))
                                    .withColumn("control_group_total_final",
                                                F.max("control_group_total").over(Window.partitionBy()))
                                )
                pivot_table = (
                                    control_freq
                                    .filter(F.col("group") != self.control_label)
                                    .drop("control_freq", "control_group_total")
                                    # categorial  total per train-test
                                    .withColumn('cat_total', F.col("freq") + F.col("control_freq_final"))
                                    .withColumn("expext_control", F.col('cat_total') * F.col("control_group_total_final") / 
                                                (F.col("group_total") + F.col("control_group_total_final")))
                                    .withColumn("expext_test", F.col('cat_total') * F.col("group_total") / 
                                                (F.col("group_total") + F.col("control_group_total_final")))
                                )
                if correction:
                    chi_components = (
                        pivot_table
                        .withColumn("chi_comp", 
                                    F.pow(F.abs(F.col("freq") - F.col("expext_test")) - 
                                        F.least(F.lit(0.5), F.abs(F.col("freq") - F.col("expext_test"))), 2) 
                                    / F.col("expext_test") + 
                                    F.pow(F.abs(F.col("control_freq_final") - F.col("expext_control")) -
                                        F.least(F.lit(0.5), F.abs(F.col("control_freq_final") - F.col("expext_control"))), 2) 
                                    / F.col("expext_control"))
                    )
                else:
                    chi_components = (
                        pivot_table
                        .withColumn("chi_comp", 
                                    F.pow(F.col("freq") - F.col("expext_test"), 2) / F.col("expext_test") + 
                                    F.pow(F.col("control_freq_final") - F.col("expext_control"), 2) / F.col("expext_control"))
                    )
                chi_square = (
                                chi_components.select(
                                    # F.col("split"),
                                    F.col("group"),
                                    # F.col("column"),
                                    (
                                        F.sum("chi_comp")
                                        .over(chi_square_window)
                                    ).alias("statistic"),
                                    F.col("group_total").alias("test_group_total"),
                                    F.col("control_group_total_final").alias("control_group_total"),
                                    F.col("category")
                                )
                                .withColumn("dof", (F.count("category").over(chi_square_window) - 1))
                                .withColumn("p_value", p_value_calc("statistic", "dof"))
                            )
                select_arr = [
                                F.col("group"),
                                F.col("statistic"),
                                F.col("p_value")
                            ]
                tmp_table = (
                                chi_square
                                .select(*select_arr)
                                .withColumn("column", F.lit(column))
                                .withColumn("split", F.lit(split))
                            )
                
                if result_table is None:
                    result_table = tmp_table
                else:
                    result_table = result_table.union(tmp_table)
        return (
                    result_table
                    .distinct()
                    .toPandas()
                )
        # return map(lambda row: row.asDict(), result_table.distinct().collect())      

    def _scipy_calc(self,
                    control_label: int, 
                    test_label: int, 
                    target_column: str, 
                    label_column: str,
                    test_function: Callable = chi2_contingency):
        control_column = np.array((
                                    self.data
                                    .filter(F.col(label_column) == control_label)
                                    .select(target_column)
                                    .fillna(np.nan)
                                    .rdd
                                    .flatMap(lambda row: row)
                                    .collect()
                                ))
        test_column =  np.array((
                                    self.data
                                    .filter(F.col(label_column) == test_label)
                                    .select(target_column)
                                    .fillna(np.nan)
                                    .rdd
                                    .flatMap(lambda row: row)
                                    .collect()
                                ))
        
        unique_values = (set(test_column) | set(control_column))

        contingency_table = np.zeros((2, len(unique_values)))
        for index, element in enumerate(unique_values):
            contingency_table[0, index] = len(control_column[control_column == element])
            contingency_table[1, index] = len(test_column[test_column == element])

        statistic, pvalue, dof, expected_freq = test_function(contingency_table)

        return {
                    "p-value": pvalue,
                    "statistic": statistic,
                    "pass": pvalue < self.reliability,
                }

    @staticmethod
    def _vectorizing_data(data: DataFrame, 
                          target_columns: Union[str, List[str]], 
                          label_column: str) -> DataFrame:
        
        if isinstance(target_columns, str):
            asembler = VectorAssembler(inputCols=[target_columns], 
                                       outputCol="features", 
                                       handleInvalid="skip")
        elif isinstance(target_columns, list):
            asembler = VectorAssembler(inputCols=target_columns, 
                                       outputCol="features",
                                       handleInvalid="skip")
        else:
            raise Exception("Unexpected target_columns!")
        return asembler.transform(data).select(label_column, "features")
    
    def _categorial_to_numeric(self, data: DataFrame) -> DataFrame:
        column_types = dict(data.dtypes)
        for index, column in enumerate(self.target_columns):
            if column_types[column] in ['string', 'varchar']:
                stringIndexer = StringIndexer(inputCol=column, 
                                              outputCol=f"{column}_indexed")
                data = stringIndexer.fit(data).transform(data)
                self.target_columns[index] = f"{column}_indexed"
        return data
    




class TestingKStest(StatTest):

    def __init__(self, data, target_columns, label_column, groups_num, k_splits, reliability=0.05, control_label = 0, realization = 1, error = 0.01):
        super().__init__(data, target_columns, label_column, groups_num, k_splits, reliability, control_label, realization)
        self.error = error

    def calculate(self) -> Dict[str, Union[float, bool]]:
        if isinstance(self.target_columns, str):
            self.target_columns = [self.target_columns]
        elif isinstance(self.target_columns, list):
            pass
        else:
            Exception(f"Something go wrong! \n {self.target_columns}")

        group_lsit = list(range(self.groups_num))
        group_lsit.pop(self.control_label)
        
        groups_pairs_list = [(self.control_label, i) for i in group_lsit]

        if self.realization and self.realization < 3: 
            statistics = self._statistic_calculation(self.data,
                                                    self.label_column,
                                                    self.target_columns,
                                                    self.groups_num,
                                                    self.k_splits,
                                                    self.control_label,
                                                    self.realization - 1)
            # print(statistics)
            
        result = {}
        for column in self.target_columns:
            tmp_dict = {}
            for split in range(self.k_splits):
                for control, test in groups_pairs_list:
                    if self.realization and self.realization < 3:
                        n_control = statistics[f"n{control}_{column}_{split}"]
                        n_test = statistics[f"n{test}_{column}_{split}"]
                        d = statistics[f"D_{test}_{column}_{split}"]
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._asymptotic_ks_pvalue(d, n_control, n_test, self.reliability)
                    elif self.realization == 3:
                        test_column, control_column = self._target_column_extracting(column, control, test)
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._alterantive_calc(test_column, control_column, column)
                    else:
                        tmp_dict[f"split: {split} groups: {control}, {test}"] = self._scipy_calc(control_label=control,
                                                                                                 test_label=test,
                                                                                                 target_column=column,
                                                                                                 label_column=self.label_column,
                                                                                                 test_function=ks_2samp)
            result[column] = tmp_dict

        return result
        
    
    @staticmethod
    def _statistic_calculation(data: DataFrame,
                               group_column: str, 
                               target_columns: List[str], 
                               groups_num: int, 
                               k_splits: int,
                               control_label: int = 0,
                               binary: bool = True,) -> DataFrame:
        final_select = []
        select_expr = []


        for split_idx in range(k_splits):
            if binary:
                group_exp = F.expr(f"ascii(substring({group_column}, {split_idx}, 1))")
            else:
                group_exp = F.col(group_column)
            for column in target_columns:
                wnd = Window.orderBy(column).rangeBetween(Window.unboundedPreceding, 0)
                wnd_total = Window.rangeBetween(Window.unboundedPreceding, Window.unboundedFollowing)

                select_expr.extend([
                    column,
                    group_column,

                    F.sum((group_exp == control_label).cast('int')).over(wnd)
                    .alias(f'# G{control_label}_{column}_{split_idx} <= x'),

                    F.sum((group_exp == control_label).cast('int')).over(wnd_total)
                    .alias(f'# G{control_label}_{column}_{split_idx}'),
                ])

                final_select.extend([
                    F.first(f'# G{control_label}_{column}_{split_idx}').alias(f'n{control_label}_{column}_{split_idx}'),
                ])

                for group in [i for i in range(groups_num) if i != control_label]:
                    select_expr.extend([
                        F.sum((group_exp == group).cast('int')).over(wnd)
                        .alias(f'# G{group}_{column}_{split_idx} <= x'),
                                                
                        F.sum((group_exp == group).cast('int')).over(wnd_total)
                        .alias(f'# G{group}_{column}_{split_idx}'),

                        F.abs((F.col(f'# G{control_label}_{column}_{split_idx} <= x') / F.col(f'# G{control_label}_{column}_{split_idx}')) -
                            (F.col(f'# G{group}_{column}_{split_idx} <= x') / F.col(f'# G{group}_{column}_{split_idx}'))).alias(f'delta_F_{group}_{column}_{split_idx}'),
                        
                    ])
                    final_select.extend([
                        F.first(f'# G{group}_{column}_{split_idx}').alias(f'n{group}_{column}_{split_idx}'),
                        F.max(f'delta_F_{group}_{column}_{split_idx}').alias(f"D_{group}_{column}_{split_idx}")
                    ])
        return (
                    data
                    .select(*select_expr)
                    .select(*final_select)
                    .collect()[0]
                )
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
    
    def _target_column_extracting(self, target_column: str, control_label: int, test_label: int) -> List[DataFrame]:
        test_column = (
                        self.data.filter(F.col(self.label_column) == test_label)
                                    .select(F.col(target_column))
                      )
        
        control_column = (
                            self.data.filter(F.col(self.label_column) == control_label)
                                        .select(F.col(target_column))
                         )

        return (test_column, control_column)

    def _alterantive_calc(self, 
                     test_column: DataFrame, 
                     control_column: DataFrame, 
                     column_name: str,) -> Dict[str, Union[float, bool]]:
        
        test_count = test_column.count()
        control_count = control_column.count()

        delta = self._error_bound(test_count, control_count) / 2

        ksi_test = delta - sqrt(delta / test_count)
        ksi_control = delta - sqrt(delta / control_count)

        sketch_test = min(ceil(1 / (delta - ksi_test) + 1), test_count)
        sketch_control = min(ceil(1 / (delta - ksi_control) + 1), control_count)

        rangs_test = np.linspace(1 / test_count, 1, sketch_test)
        rangs_control = np.linspace(1 / control_count, 1, sketch_control)

        F_test = test_column.approxQuantile(column_name, 
                                            list(rangs_test), 
                                            ksi_test)

        F_control = control_column.approxQuantile(column_name, 
                                                  list(rangs_control), 
                                                  ksi_control)
        
        return self._scipy_cdf(F_test, F_control, self.reliability)
    
    def _error_bound(self, n: int, m: int) -> float:
        alpha_up = self.reliability + self.error
        alpha_down = self.reliability - self.error
        alpha = self.reliability
        
        crit_D_alpha = self._D_crit(n, m, alpha)
        crit_D_alpha_up = self._D_crit(n, m, alpha_up)
        crit_D_alpha_down = self._D_crit(n, m, alpha_down)

        return min(abs(crit_D_alpha - crit_D_alpha_down), abs(crit_D_alpha - crit_D_alpha_up))
    
    @staticmethod
    def _scipy_cdf(F_1 : List[float], F_2: List[float], reliability: float) -> dict:
        statistic, p_value = ks_2samp(F_1,
                                      F_2,
                                      method='asymp')
        
        return {
                    "p-value": p_value,
                    "statistic": statistic,
                    "pass": p_value < reliability,
                } 

    @staticmethod
    def _D_crit(n: int, m: int, alpha: float) -> float:
        c_alpha = sqrt(-log(alpha / 2) * 0.5)
        multiplier = sqrt((n + m) / (n * m))

        return c_alpha * multiplier
    
    