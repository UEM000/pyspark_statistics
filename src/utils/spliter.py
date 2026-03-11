from __future__ import annotations

from .enums import DataRealization

from typing import List, Optional, Union

from pyspark.sql import DataFrame, types
import pyspark.sql.functions as F
import numpy as np
import pandas as pd
import polars as pl

class Spliter():
    """
    Однократное разбиение на группы
    """

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
    
    def __init__(self,
                 data: Union[DataFrame, pd.DataFrame],
                 random_state: int = 21,
                 groups_num: int = 2,
                 fractions: List[float]=None):

        self.data = data
        self.random_state = random_state
        self.groups_num = groups_num
        self.data_realization = self._select_data_realization(data)

        if fractions is not None:
            self.fractions = fractions
        else:
            self.fractions = [(1 / groups_num) for _ in range(groups_num)]

        if self.data_realization == DataRealization.pandas:
            self.size = len(self.data)
        elif self.data_realization == DataRealization.polars:
            if isinstance(self.data, pl.DataFrame):
                self.size = self.data.shape[0]
                self._is_lazy = False
            else:
                self.size = self.data.select(pl.len()).collect().item()
                self._is_lazy = True

    def _new_split_gen(self, number_of_generation=0, return_char=True):
        _rand_col = F.rand(seed=self.random_state + number_of_generation)
        new_split = F
        for _i, _th in enumerate(np.cumsum(self.fractions)):
            new_split = new_split.when(_rand_col <= _th, _i)
        if return_char:
            new_split = F.char(new_split.cast("int"))
        return new_split

class StandartSpliter(Spliter):

    def __init__(self,
                 data,
                 random_state = 21,
                 groups_num = 2):
        super().__init__(data, random_state, groups_num)


    def split(self, number_of_generation: int=0) -> Union[DataFrame, pd.DataFrame]:
        if self.data_realization == DataRealization.spark:
            df_with_groups = (
                self.data
                .withColumn("group", self._new_split_gen(number_of_generation=number_of_generation, return_char=False).cast("int"))
            )
        elif self.data_realization == DataRealization.pandas:
            np.random.seed(self.random_state + number_of_generation)
            df_with_groups = (
                self.data
                .assign(group=np.random.choice(list(range(self.groups_num)), size=self.size, p=self.fractions))
            )
        elif self.data_realization == DataRealization.polars:
            np.random.seed(self.random_state + number_of_generation)
            # Оптимизация: используем polars.int_range для генерации индексов групп
            # Это быстрее чем numpy.random.choice для больших данных
            group_indices = np.random.choice(
                list(range(self.groups_num)), 
                size=self.size, 
                p=self.fractions
            )

            if self._is_lazy:
                df_with_groups = (
                    self.data
                    .with_columns(group=pl.Series("group", group_indices).cast(pl.Int16))
                )
            else:
                df_with_groups = (
                    self.data
                    .with_columns(group=pl.Series("group", group_indices, dtype=pl.Int16))
                )

        return df_with_groups

class BinarySpliter(Spliter):
    """
    Множественное разбиение на группы
    """

    def __init__(self, data, random_state = 21, groups_num = 2, fractions = None, k_splits: int=1):
        super().__init__(data, random_state, groups_num, fractions)
        self.k_splits = k_splits
        

        
    
    def _new_split_gen(self, number_of_generation=0, return_char=True):
        _rand_col = F.rand(seed=self.random_state + number_of_generation)
        new_split = F
        for _i, _th in enumerate(np.cumsum(self.fractions)):
            new_split = new_split.when(_rand_col <= _th, _i)
        if return_char:
            new_split = F.char(new_split.cast("int"))
        return new_split
    
    def split(self) -> Union[DataFrame, pd.DataFrame]:

        if self.data_realization == DataRealization.spark:
            df_with_groups = (
                                self.data
                                .withColumn(
                                    'group',
                                    F.concat(*[
                                        self._new_split_gen(number_of_generation=_number_of_generation)
                                        for _number_of_generation in range(self.k_splits)
                                    ]).cast("binary")
                                )
                            )
        elif self.data_realization == DataRealization.pandas:
            np.random.seed(self.random_state)
            df_with_groups = self.data.assign(group=np.random.choice(
                                                                        list(range(self.groups_num)), 
                                                                        size=(self.size, self.k_splits), 
                                                                        p=self.fractions
                                                                    ).tolist())
        else:
            raise TypeError("No polars realization!")

        return df_with_groups