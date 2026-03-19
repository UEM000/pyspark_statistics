import numpy as np
import pyspark.sql as spark
import pyspark.sql.functions as F
import faiss

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark import StorageLevel, Broadcast, RDD
from collections import defaultdict
from typing import (
    Union,
    List,
    Literal,
    Optional,
    Iterable,
    Tuple,
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

    Modes
    -----
        **base**        : Собирает все данные  на драйвере и строит один faiss индекс.
                        Используется только при небольшом количестве данных.
        
        **fast**        : Предварительная кластеризация данных и построение индекса локально
                        на ближайшем кластере. Используется для больших датасетов.
        
        **partition**   : обучаем IVF на сэмпле из данных. Далее строим для каждой партиции
                          индекс и после локального поиска соседей в партиции, выбираем
                          среди лучших самые ближайшие. Используется для больших датасетов.

        **auto**        : автоматический выбор режима. 
                        По-умолчанию `fast`.
    """
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK
    _SAMPLE_TARGET = 5_000_000

    def __init__(self,
                 n_neighbors: int = 1,
                 k: int = 1000,
                 seed: Union[int, None] = None,
                 feature_cols: Union[List[str], None] = None,
                 faiss_mode: Literal["base", "fast", "partition", "auto"] = "auto",
                 ):
        self.n_neighbors = n_neighbors
        self.k = k 
        self.seed = seed or 21
        self.feature_cols = feature_cols
        self.faiss_mode = faiss_mode

        self._kmeans_model = None
        self._index = None
        self._id_map: Optional[np.ndarray] = None
        self._centroid_index: Optional[faiss.Index] = None
        self._centroid_list: Optional[List[np.ndarray]] = None
        self._clustered_data: Optional[spark.DataFrame] = None
        self._mode: Optional[Literal["base", "fast"]] = None
        self._sharded_rdd: Optional[RDD] = None

    def _vectorize_data(self, data: spark.DataFrame) -> spark.DataFrame:
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

    def _direct_fit(self, data: spark.DataFrame) -> None:
        """
        Прямое вычисление faiss с выгрузкой данных на драйвер.

        Args
        ----
            data: `spark.DataFrame`
                Данные для которых мы ищем соседей.
        """
        prepeared_data = self._vectorize_data(data)
        select_cols = ["_id", "features"]
        rows = prepeared_data.select(*select_cols).collect()
        X = np.array([list(row.features) for row in rows], dtype=np.float32)
        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(X)

    def _clustered_fit(self, data: spark.DataFrame) -> None:
        """
        Кластеризация данных спомощью Spark KMeans и построение
        индекса центроидов на драйвере. Центроиды будут использоваться
        при поиске соседей для запросов для уменьшения потребления памяти.

        Args
        ----
            data: `spark.DataFrame`
                Данные для которых мы ищем соседей.
        """
        prepeared_data = self._vectorize_data(data)
        df_clustered = self._clustering(prepeared_data)
        select_cols = ["_id", "cluster_id", "features"]
        self._clustered_data = (
            df_clustered
            .select(*select_cols)
            .persist(self.PERSIST_POLITIC)
        )
        self._clustered_data.count()
        
        X = np.array(self._centroid_list, dtype=np.float32)
        self._centroid_index= faiss.IndexFlatL2(X.shape[1])
        self._centroid_index.add(X)

    def _partition_fit(self, data: spark.DataFrame) -> None:
        """
        Реализация partition faiss fit. 
        Предполагается, что будет браться sample данных, 
        который отражает св-ва основной выборки.
        Для этого стоит, вообще говоря, проводить АА-тест.
        Здесь будем использовать обычный sample для удобства.

        Args
        ----
            data: `spark.DataFrame`
                Данные для которых мы ищем соседей.
        
        Return
        ------
            None
        """
        prepeared_data = self._vectorize_data(data)
        self._clustered_data = prepeared_data
        session = data.sparkSession

        data_size = prepeared_data.count()
        frac = min(self._SAMPLE_TARGET / max(data_size, 1), 1.0)

        sample_rows = (
                        prepeared_data
                        .sample(fraction=frac, seed=self.seed)
                        .select("features")
                        .collect()
                    )
        X = np.array(
            [list(row['features']) for row in sample_rows],
            dtype=np.float32,
        )

        d = X.shape[1]
        # IVF Faiss подерживает до 39 * (training points) на один кластер
        nlist = min(self.k, max(1, X.shape[0] // 39)) 

        quantizer = faiss.IndexFlatL2(d)
        self._index = faiss.IndexIVFFlat(quantizer, d, self.k)
        self._index.train(X)

        self._index = faiss.serialize_index(self._index)
        bc_index = session.sparkContext.broadcast(self._index)

        features = ["_id", "features"]
        self._sharded_rdd = (
            prepeared_data
            .select(*features)
            .rdd
            .mapPartitions(lambda it: FaissSpark._patition_faiss(it, bc_index))
        )

    @staticmethod
    def _patition_faiss(iterator: Iterable, bc_index: Broadcast):
        """
        Fit на локально на каждой партиции на осонвании данных из sample-а.

        Args
        ----
        """
        index = faiss.deserialize_index(bc_index.value)

        ids, vectors = [], []
        for row in iterator:
            ids.append(row["_id"])
            vectors.append(list(row['features']))
        
        if not ids:
            return # для случая пустой партиции

        ids = np.array(ids, dtype=np.int64)
        vectors = np.array(vectors, dtype=np.float32)

        index_with_ids = faiss.IndexIDMap(index)
        index_with_ids.add_with_ids(vectors, ids)

        # share_bytes = faiss.serialize_index(index_with_ids)
        # yield (share_bytes, )
        yield faiss.serialize_index(index_with_ids)

    def _direct_predict(self, test_data: spark.DataFrame) -> List[List[Union[int, float]]]:
        """
        Нахождение индексов прямым методом faiss.
        """
        rows = test_data.select("features").collect()
        X = np.array([list(row.features) for row in rows], dtype=np.float32)
        dist, pos_indexes = self._index.search(X, k=self.n_neighbors)
        
        # Возвращаем список кортежей: [(query_idx, neighbor_idx, distance), ...]
        result = []
        for query_idx in range(len(X)):
            for i in range(self.n_neighbors):
                pos = int(pos_indexes[query_idx][i])
                original_id = int(self._id_map[pos]) if pos >= 0 else -1
                distance = float(dist[query_idx][i])
                result.append((query_idx, original_id, distance))
                # neighbor_idx = int(indexes[query_idx][i])
                # distance = float(dist[query_idx][i])
                # result.append((query_idx, neighbor_idx, distance))
        
        return result
    
    def _clustering_predict(self, test_data: spark.DataFrame) -> List[List[Union[int, float]]]:
        """
        Нахождение индексов с помощью кластеризации.
        """
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
                for i in range(k):
                    neighbor_idx = int(r_indexes[0][i])
                    distance = float(r_dist[0][i])
                    result.append((query_idx, neighbor_idx, distance))
        
        return result
    
    def _partition_predict(self, test_data: spark.DataFrame) -> List[List[Union[int, float]]]:
        """
        Реализация partition faiss predict.

        Args
        ----
            test_data : `spark.DataFrame`
                Данные для которых мы ищем соседей.

        Return
        ------
            None
        """
        rows = test_data.select("features").collect()
        session = self._clustered_data.sparkSession
        X = np.array([list(row['features']) for row in rows], dtype=np.float32)
        results = []

        for query_idx, query in enumerate(X): 
            # print(query)        
            bc_query = session.sparkContext.broadcast(query)
            tmp_result = np.array(
                self._sharded_rdd
                .mapPartitions(lambda it:
                               FaissSpark._per_partition_predict(it, bc_query))
                .flatMap(lambda x: x)
                .collect()
            )
            bc_query.unpersist()
            results.extend([(query_idx, idx, dist) 
                            for dist, idx in tmp_result[tmp_result[:, 0].argsort()[: self.n_neighbors]]])
        return results

    @staticmethod
    def _per_partition_predict(shard_iter: Iterable, 
                               query: Broadcast, 
                            #    n_neighbors: int
    ):
        """
        Predict локально на каждой партиции.

        Args
        ----
            shard_iter: `Iterable`
                Итератор с шардами индексов.

            query: `np.ndarray`
                Вектор запроса (передаётся напрямую, сериализуется корректно).

            n_neighbors: `int`
                Количество соседей для поиска.
        """
        query = np.array(query.value, dtype=np.float32).reshape(1, -1)
        for shard in shard_iter:
            index = faiss.deserialize_index(shard)
            k = min(10, index.ntotal)
            distances, ids = index.search(query, k)
            yield [
                (float(distances[0][i]), int(ids[0][i]))
                for i in range(k)
                if ids[0][i] >= 0
            ]
            
    def _clustering(self, data: spark.DataFrame) -> spark.DataFrame:
        """
        Разбиение на кластеры для дальнейшего использования faiss для каждого кластера.

        Agrs
        ----
            data : `SparkDataFrame`
                Подготовленные данные с помощью _vectorize_data
        
        Returns
        -------
            df_clustered : `SparkDataFrame`
                Датафрейм с колонкой пренадлежности к кластеру.
        """
        # vectorized_data = self._vectorize_data(data)
        self._kmeans_model = KMeans(
            k=self.k, 
            seed=self.seed,
            featuresCol="features",
            predictionCol='cluster_id'
        )
        self._kmeans_model = self._kmeans_model.fit(data)
        df_clustered = self._kmeans_model.transform(data)
        self._centroid_list = self._kmeans_model.clusterCenters()

        return df_clustered

    def _parametrs_tuning(self):
        """
        Тюнинг параметров.
        """
          

    def _calculation_mode(self, count: int) -> Literal["base", "fast", "partition"]:
        """
        Выбор типа вычисления faiss.

        Agrs:
        -----
            count: `int`
                Количество строка в датафрейме.
        """
        if self.faiss_mode in ("base", "fast", "partition"):
            return self.faiss_mode 
        
        if count > 1_000_000:
            return "fast"
        return "base"
    
    def fit(self, data: spark.DataFrame) -> "FaissSpark":
        """
        Fit.

        Args
        ----
            data: `spark.DataFrame`
                Данные из которых тянутся соседи.
        """
        count = data.count()
        self._mode = self._calculation_mode(count)

        if self._mode == "base":
            self._direct_fit(data)
        elif self._mode == "fast":
            self._clustered_fit(data)
        elif self._mode == "partition":
            self._partition_fit(data)

        return self
    
    def predict(
            self, test_data: spark.DataFrame
    ) -> List[Tuple[int, int, float]]:
        """
        Predict.
        
        Returns
        -------
        result : `List[Tuple[int, int, float]]`
            Список кортежей: (порядковый номер записи в test_data, найденный близнец, расстояние до него)
        """
        prepeared_data = self._vectorize_data(test_data)

        if self._mode == "base":
            result = self._direct_predict(prepeared_data)
        elif self._mode == "fast":
            result = self._clustering_predict(prepeared_data)
        elif self._mode == "partition":
            result = self._partition_predict(prepeared_data)
        else:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")
        
        return result

    def unpersist(self) -> None:
        """
        Подчистить ресурсы spark.
        """
        if self._clustered_data is not None:
            self._clustered_data.unpersist()
            self._clustered_data = None
    
    def __enter__(self) -> "FaissSpark":
        return self
        
    def __exit__(self, *_) -> None:
        self.unpersist()
            