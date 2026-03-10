import numpy as np
import pyspark.sql as spark
import pyspark.sql.functions as F
import faiss

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark import StorageLevel

from collections import defaultdict
from typing import (
    Union,
    List,
    Literal,
    Optional
)
class FaissSpark:
    """
    Реализация faiss на pyspark:
    ---------
        * **Большие данные** (> 1_000_000):
            1. Кластеризация данных;
            2. Используем faiss для кластера, а не для всего датасета.
        * **Маленькие данные**:
            1. Преобрауем датасет в pandas;
            2. Используем faiss на нашем датасете.
    """
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK

    def __init__(self,
                 n_neighbors: int = 1,
                 k: int = 1000,
                 seed: Union[int, None] = None,
                 feature_cols: Union[List[str], None] = None,
                 faiss_mode: Literal["base", "fast", "auto"] = "auto",
                 ):
        self.n_neighbors = n_neighbors
        self.k = k 
        self.seed = seed
        self.feature_cols = feature_cols
        self.faiss_mode = faiss_mode

        self._kmeans_model = None
        self._index = None
        self._centroid_index = None
        self._centroid_list: List[np.ndarray] | None = None
        self._clustered_data: spark.DataFrame | None = None
        self._mode: Optional[Literal["base", "fast"]] = None

    def _vectorize_data(self, data: spark.DataFrame) -> spark.DataFrame:
        """
        Подготовка входных данных: векторизация и проверка на категориальные фичи.
        Все незакодированные фичи будут вызывать ошибку работы / (в дальнейшем просто выкидываться?).

        Parameters
        ----------
            data : SparkDataFrame
                Входные данные. Должен содержать:
                    - Числовые фичи;
                    - Заэнкоженные категориальные фичи;
        
        Returns
        -------
            vectorized_data: SparkDataFrame
                входной датасет с колонкой векторов из фичей.
        """
        if self.feature_cols is None:
            self.feature_cols = data.columns
        if len(set(map(lambda x: x[1], data.dtypes)).intersection(['varchar', 'string'])) > 0:
            raise TypeError("Unencoded categorical features are not allowed!")

        vecAssembler = VectorAssembler(inputCols=self.feature_cols,
                                     outputCol="features",
                                     handleInvalid="keep")
        
        return (
                    vecAssembler
                    .transform(data)
                    # .select("features")
                )

    def _direct_fit(self, data: spark.DataFrame) -> None:
        """
        Прямое вычисление faiss с выгрузкой данных на драйвер.
        """
        prepeared_data = self._vectorize_data(data)
        select_cols = ["features"]
        rows = prepeared_data.select(*select_cols).collect()
        X = np.array([list(row.features) for row in rows], dtype=np.float32)
        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(X)

    def _clustered_fit(self, data: spark.DataFrame) -> None:
        """
        Вычисление faiss с предварительной кластеризацией
        """
        prepeared_data = self._vectorize_data(data)
        df_clustered = self._clustering(prepeared_data)
        select_cols = ["cluster_id", "features"]
        self._clustered_data = (
            df_clustered
            .select(*select_cols)
            .persist(self.PERSIST_POLITIC)
        )
        self._clustered_data.count()
        
        X = np.array(self._centroid_list, dtype=np.float32)
        self._centroid_index= faiss.IndexFlatL2(X.shape[1])
        self._centroid_index.add(X)

    def _direct_predict(self, test_data: spark.DataFrame) -> List[List[Union[int, float]]]:
        rows = test_data.select("features").collect()
        X = np.array([list(row.features) for row in rows], dtype=np.float32)
        dist, indexes = self._index.search(X, k=self.n_neighbors)
        
        # Возвращаем список кортежей: [(query_idx, neighbor_idx, distance), ...]
        result = []
        for query_idx in range(len(X)):
            for i in range(self.n_neighbors):
                neighbor_idx = int(indexes[query_idx][i])
                distance = float(dist[query_idx][i])
                result.append((query_idx, neighbor_idx, distance))
        
        return result
    
    def _clustering_predict(self, test_data: spark.DataFrame) -> List[List[Union[int, float]]]:
        result = []
        rows = test_data.select("features").collect()
        X = np.array([list(row.features) for row in rows], dtype=np.float32)
        _, cluster_ids = self._centroid_index.search(X, k=1)
        # В spark KMeans центроиды возвращаются в соответствии с порядковым номером их кластера
        # Поэтому индекс сразу указывает нам на то, какой кластер нам нужен
        cluster_to_queries = defaultdict(list)
        for query_idx, cluster_idx in enumerate(cluster_ids.flatten()):
            cluster_to_queries[cluster_idx].append(query_idx)

        for cluster_idx, query_idx_list in cluster_to_queries.items():
            cluster_rows = (
                self._clustered_data
                .filter(F.col('cluster_id') == cluster_idx)
                .select("features")
                .collect()
            )
            
            # Проверка на пустой кластер
            if len(cluster_rows) == 0:
                for query_idx in query_idx_list:
                    result.append((query_idx, -1, float('inf')))
                continue
            
            cluster_data = np.array(
                [list(row.features) for row in cluster_rows],
                dtype=np.float32
            )
            tmp_index = faiss.IndexFlatL2(cluster_data.shape[1])
            tmp_index.add(cluster_data)
            k = min(self.n_neighbors, len(cluster_data))
            for query_idx in query_idx_list:
                r_dist, r_indexes = tmp_index.search(X[query_idx: query_idx+1], k=k)
                # Возвращаем результат: (query_idx, neighbor_idx, distance)
                for i in range(k):  # ✅ Используем k, а не self.n_neighbors
                    neighbor_idx = int(r_indexes[0][i])
                    distance = float(r_dist[0][i])
                    result.append((query_idx, neighbor_idx, distance))
        
        return result
        

    def _clustering(self, data: spark.DataFrame) -> spark.DataFrame:
        """
        Разбиение на кластеры для дальнейшего использования faiss для каждого кластера.

        Parametrs
        ----------
            data : SparkDataFrame
                Подготовленные данные с помощью *_vectorize_data*
        
        Returns
        -------
            df_clustered : SparkDataFrame
                Датафрейм с колонкой пренадлежности к кластеру.
        """
        # vectorized_data = self._vectorize_data(data)
        self._kmeans_model = KMeans(k=self.k, 
                                    seed=self.seed,
                                    featuresCol="features",
                                    predictionCol='cluster_id')
        self._kmeans_model = self._kmeans_model.fit(data)
        df_clustered = self._kmeans_model.transform(data)
        self._centroid_list = self._kmeans_model.clusterCenters()

        return df_clustered
    
    def _calculation_mode(self, count: int):
        """
        Выбор типа вычисления faiss.
        """
        if self.faiss_mode == "base":
            return "base"
        if self.faiss_mode == "fast":
            return "fast"
        if self.faiss_mode == "auto":
            if count > 1_000_000:
                return "fast"
            return "base"
    
    def fit(self, data: spark.DataFrame) -> "FaissSpark":
        """
        Fit.
        """
        count = data.count()
        self._mode = self._calculation_mode(count)

        if self._mode == "base":
            self._direct_fit(data)
        if self._mode == "fast":
            self._clustered_fit(data)

        return self
    
    def predict(self, test_data: spark.DataFrame):
        """
        Predict.
        
        Returns
        -------
        result : List[Tuple[int, int, float]]
            Список кортежей: (порядковый номер записи в test_data, найденный близнец, расстояние до него)
        """
        prepeared_data = self._vectorize_data(test_data)
        if self._mode == "base":
            result = self._direct_predict(prepeared_data)
        elif self._mode == "fast":
            result = self._clustering_predict(prepeared_data)
        else:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")
        
        return result

    def unpersist(self):
        """
        Подчистить ресурсы spark.
        """
        if self._clustered_data is not None:
            self._clustered_data.unpersist()
            self._clustered_data = None
        
    def __del__(self):
        self.unpersist()
            