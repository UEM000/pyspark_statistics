import faiss
import numpy as np
import pyspark.sql as spark
import pandas as pd

from pyspark.ml.feature import VectorAssembler
from pyspark import StorageLevel, Broadcast, RDD
import pyspark.sql.functions as F

from typing import (
    Union,
    List,
    Iterator,
)
from collections import defaultdict

class FaissBenchmark:
    """
    Бэнчмарк для поиска соседей small-data vs big-data.
    """

    def __init__(
            self,
            n_neighbors: int = 1,
            feature_cols: Union[List[str], None] = None,
            batch_size: int = 512
    ):
        """
        Инициализация бенчмарка FAISS.

        Args
        ----------
            n_neighbors : `int`
                Количество соседей для поиска.

            feature_cols : `Union[List[str], None]`
                Список колонок для векторизации. Если `None`, используются все колонки.

            batch_size : `int`
                Размер батча для обработки (зарезервировано).
        """
        self.n_neighbors  = n_neighbors
        self.feature_cols = feature_cols
        self.batch_size   = batch_size
    
    def _vectorize_data(
            self, 
            data: spark.DataFrame
    ) -> spark.DataFrame:
        """
        Подготовка входных данных: векторизация и проверка на категориальные фичи.
        Все незакодированные фичи будут вызывать ошибку работы / (в дальнейшем просто выкидываться?).

        Agrs
        ----------
            data : `SparkDataFrame`
                Входные данные. Должен содержать:
                    - Числовые фичи;
                    - Заэнкоженные категориальные фичи;
        
        Returns
        -------
            vectorized_data: `SparkDataFrame`
                входной датасет с колонкой векторов из фичей.
        """
        if self.feature_cols is None:
            self.feature_cols = data.columns
        if len(set(map(lambda x: x[1], data.dtypes)).intersection(['varchar', 'string'])) > 0:
            raise TypeError("Unencoded categorical features are not allowed!")

        vecAssembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features",
            handleInvalid="keep"
        )
        
        return (
                    vecAssembler
                    .transform(data)
                    .withColumn('_id', F.monotonically_increasing_id()) # Колонка с уникальным идентификатором строки
                )

    @staticmethod
    def _local_index_bulder(
        it: Iterator,
        bc_vectors: Broadcast,
        bc_n_neighbors: Broadcast,
        # bc_batch_size: Broadcast
    ):
        """
        Построение локального FAISS индекса на партиции и поиск ближайших соседей.

        На каждой партиции train_data строится IndexIDMap (IndexFlatL2 + ID),
        после чего выполняется поиск ближайших соседей для всех векторов из test_data.

        Args
        ----------
            it : `Iterator`
                Итератор по строкам партиции train_data.
                Должен содержать колонки `_id` и `features`.

            bc_vectors : `Broadcast`
                Заброадкащенный массив векторов test_data (np.ndarray, dtype=np.float32).

            bc_n_neighbors : `Broadcast`
                Заброадкащенное количество соседей для поиска.

        Yields
        ------
            batch : `List[List[Tuple[int, int, float]]]`
                Список списков кортежей:
                    - query_idx : индекс запроса из test_data
                    - neighbor_id : `_id` найденного соседа из train_data
                    - distance : расстояние (L2) до соседа
        """
        import faiss
        import numpy as np

        n_neighbors = bc_n_neighbors.value
        rows = []
        ids = []
        for row in it:
            rows.append(row.features.toArray())
            ids.append(row._id)
        
        rows = np.array(rows, dtype=np.float32)
        ids = np.array(ids, dtype=np.int64)

        index = faiss.IndexFlatL2(rows.shape[1])
        index = faiss.IndexIDMap(index)
        index.add_with_ids(rows, ids)

        batch = []
        dist, pos_indexes = index.search(bc_vectors.value, n_neighbors)

        for query_idx in range(len(bc_vectors.value)):
            tmp_list = []
            for i in range(n_neighbors):
                faiss_returned_id = int(pos_indexes[query_idx][i])
                distance = float(dist[query_idx][i])
                if faiss_returned_id != -1:
                    neighbor_id = faiss_returned_id
                    tmp_list.append((query_idx, neighbor_id, distance))  

            if tmp_list:
                batch.append(tmp_list)

        if batch:
            yield batch    

    def search(
            self,
            session: spark.SparkSession,
            train_data: spark.DataFrame,
            test_data: spark.DataFrame
    ) -> pd.DataFrame:
        """
        Поиск top-k ближайших соседей в train_data для каждого элемента test_data.

        Алгоритм
        --------
            1. Векторизует train_data и test_data;
            2. Тестовые векторы бродкастятся на все executor'ы;
            3. На каждой партиции train_data строится локальный FAISS-индекс;
            4. Для каждого test-вектора ищутся k ближайших соседей;
            5. Результаты собираются на драйвере, мерджатся и возвращаются как DataFrame.

        Args
        ----------
            session : `SparkSession`
                Активная Spark-сессия.

            train_data : `SparkDataFrame`
                Данные, в которых ищем соседей (должен содержать числовые фичи).

            test_data : `SparkDataFrame`
                Данные, для которых ищем соседей.

        Returns
        -------
            result : `pd.DataFrame`
                Таблица с колонками:
                    - index : порядковый номер запроса из test_data
                    - n_id : `_id` найденного соседа из train_data
                    - dist : расстояние (L2) до соседа
        """
        trasformed_data = self._vectorize_data(train_data)
        transformed_test_data = self._vectorize_data(test_data)
        shard_rdd = (
            trasformed_data
            .select("_id", "features")
            .rdd
        )
        test = np.array([list(row.features) for row in transformed_test_data.collect()], dtype=np.float32)

        bc_vectors = session.sparkContext.broadcast(test)
        bc_n_neighbors = session.sparkContext.broadcast(self.n_neighbors)
        # bc_batch_size = session.sparkContext.broadcast(self.batch_size)

        shard_rdd = shard_rdd.mapPartitions(lambda it: 
                                                FaissBenchmark._local_index_bulder(
                                                    it=it, 
                                                    bc_vectors=bc_vectors,
                                                    bc_n_neighbors=bc_n_neighbors,
                                                    # bc_batch_size=bc_batch_size
                                                )
                                            )
        
        result = defaultdict(list)
        for batch in shard_rdd.toLocalIterator():
            for query_idx, idx_local_top in enumerate(batch):
                for element in idx_local_top:
                    result[query_idx].append((element[1], element[2]))

        for key in result.keys():
            result[key] = sorted(result[key], key=lambda x: x[1])[:self.n_neighbors]

        result = pd.Series(result, name="result").explode().reset_index()
        result[["n_id", "dist"]] = result['result'].apply(pd.Series)
        result = result.drop(columns=['result'])
        return result