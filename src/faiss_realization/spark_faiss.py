import numpy as np
import pyspark.sql as spark
import faiss

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark import StorageLevel

from typing import (
    Union,
    List,
    Literal
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
    PERSIST_POLITIC = StorageLevel.DISK_ONLY

    def __init__(self,
                #  data: spark.DataFrame,
                 n_neighbors: int = 1,
                 k: int = 1000,
                 seed: Union[int, None] = None, 
                 feature_cols: Union[List[str], None] = None,
                 faiss_mode: Literal["base", "fast", "auto"] = "auto",
                 mode: Literal["auto", "fit", "predict"] = "auto"
                 ):
        # self.data = data
        self.n_neighbors = n_neighbors
        self.k = k
        self.seed = seed
        self.feature_cols = feature_cols
        self.faiss_mode = faiss_mode
        
        self._kmeans_model = None
        self._index = None
        self._centroid_index = None
        self._centroid_list: List[np.ndarray] = None
        self._clustered_data: spark.DataFrame = None
        self._mode = None

    def _vectorize_data(self, data: spark.DataFrame) -> spark.DataFrame:
        """
        Подготовка входных данных: векторизация и проверка на категориальные фичи.
        Все незакодированные фичи будут вызывать ошибку работы / (в дальнейшем просто выкидываться?).

        Параметры
        ----------
            data : *SparkDataFrame*
                Входные данные. Должен содержать:
                    - Числовые фичи;
                    - Заэнкоженные категориальные фичи;
        
        Возвращает
        ----------
            *SparkDataFrame* 
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
        rows = data.collect()
        X = np.array([list(row.features) for row in rows], dtype=np.float32)
        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(X)

    def _clustered_fit(self, data: spark.DataFrame) -> None:
        """
        Вычисление faiss с предварительной кластеризацией
        """
        prepeared_data = self._vectorize_data(data)
        df_clustered = self._clustering(prepeared_data)
        self._clustered_data = df_clustered.persist(self.PERSIST_POLITIC)
        X = np.array(self._centroid_list, dtype=np.float32)
        self._idnex= faiss.IndexFlatL2(X.shape[1])
        self._index.add(X)

    def _direct_predict(self, test_data: spark.DataFrame):
        rows = test_data.collect()
        X = np.array([list(row) for row in rows], dtype=np.float32)
        dist, indexes = self._index.search(X, k=self.n_neighbors)
    
    def _clustering_predict(self, test_data: spark.DataFrame):
        ...

    def _clustering(self, data: spark.DataFrame) -> spark.DataFrame:
        """
        Разбиение на кластеры для дальнейшего использования faiss для каждого кластера.

        Параметры
        ----------
            data : *SparkDataFrame*
                Подготовленные данные с помощью *_vectorize_data*
        
        Возвращает
        ----------
            *SparkDataFrame* 
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
        """
        count = data.count()
        self._mode = self._calculation_mode(count)

        if self._mode == "base":
            self._direct_fit(data)
        if self._mode == "fast":
            self._clustered_fit(data)

        return self
    
    def predict(self, test_data: spark.DataFrame) -> spark.DataFrame:
        """
        """
        if self._mode == "base":
            result = self._direct_predict(test_data)
        if self._data == "fast":
            result = self._clustering_predict(test_data)

        
            