from .faiss_config import test_list, experiment_data
from ..faiss_realization.spark_faiss import FaissSpark

from pyspark.sql import SparkSession
import pyspark.sql as spark

from typing import (
    Dict,
    List,
    Any,
)

class FaissExperiment:
    
    def __init__(self,
                 data: spark.DataFrame,
                 report: str=None,
                 output: str=None):
        self.data = data
        self.report = report
        self.output = output

        self.test_list: List[Dict[str, Dict[str, List[int]]]] = test_list
        self.constants: Dict[str, Any] = experiment_data

    def execute(self):
        for tests in self.test_list:
            ...

    def _compare_approaches(self, 
                            train_data: spark.DataFrame,
                            test_data: spark.DataFrame,
                            current_params: Dict[str, Any]):
        """
        Сравнение обычного Faiss vs Clustered Faiss.

        Parameters
        ----------
            train_data: spark.DataFrame
                Датасет, в котором ищем похожие наблюдения;
            test_data: spark.DataFrame
                Наблюдения, для которых ищем похожие из обучающих данных;
            current_params: Dict[str, Any]
                Параметры для нахождения похожих векторов.
        """

        fast_matching = FaissSpark(
            n_neighbors=current_params['n_neighbors'],
            k=current_params['k'],
            faiss_mode='fast'
        )
        fast_matching.fit(train_data)
        fast_result = fast_matching.predict(test_data)

        direct_matching = FaissSpark(
            n_neighbors=current_params['n_neighbors'],
            k=current_params['k'],
            faiss_mode='base'
        )
        direct_matching.fit(train_data)
        direct_results = direct_matching.predict(test_data)

        # Пока не понятно, какую метрику использовать для сравнения, 
        # чтобы она в полной мере характериховала точность расчетов.
        
    def _write_output(self):
        ...

    def _write_report(self):
        ...