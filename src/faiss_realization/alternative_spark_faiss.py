from __future__ import annotations
import numpy as np
import pyspark.sql as spark
import pyspark.sql.functions as F
import faiss
import os
import gc
import sys

from pyspark.ml.feature import VectorAssembler
from pyspark import StorageLevel, Broadcast, RDD
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, 
    StructField, 
    LongType,
    FloatType
)
from sklearn.cluster import MiniBatchKMeans, Birch
from typing import (
    Union,
    List,
    Literal,
    Optional,
    Iterable,
    Tuple,
    Any
)
from tqdm import tqdm
from index_cacher import SessionError
            
    

class FaissSpark:
    """
    Реализация faiss на pyspark:
    ---------
        * **Большие данные**:
            1. Обучаем IVF на сэмпле из данных;
            2. Строим для каждой партиции индекс;
            3. После локального поиска соседей в партиции выбираем
               среди лучших самые ближайшие.
        * **Маленькие данные**:
            1. Собирает все данные на драйвере и строит один faiss индекс.

    Modes
    -----
        **base**        : Собирает все данные на драйвере и строит один faiss индекс.
                        Используется только при небольшом количестве данных.

        **partition**   : Обучаем IVF на сэмпле из данных. Далее строим для каждой партиции
                          индекс и после локального поиска соседей в партиции, выбираем
                          среди лучших самые ближайшие. Используется для больших датасетов.

        **auto**        : автоматический выбор режима.
                        По-умолчанию `partition`.

    Realization
    -----------
    fit() на train_data
    └─> _partition_fit()
        └─> На КАЖДОМ executor'е строится IndexIDMap (IVFFlat + векторы партиции)
            └─> Сериализуется в байты через faiss.serialize_index()
                └─> RDD персистится в памяти

    predict() на test_data
    └─> _partition_predict()
        ├─> toLocalIterator() — ПОСЛЕДОВАТЕЛЬНО забираем байты с executor'ов на драйвер
        │   ├─> deserializable в FAISS объект
        │   ├─> Запись в файл .index на драйвере
        │   └─> sparkContext.addFile() — рассылает на ВСЕ executor'ы
        │
        └─> mapPartitions на test_data:
            └─> На КАЖДОМ executor'е test_data:
                ├─> Для каждого батча запросов:
                │   └─> Для КАЖДОГО .index файла (все партиции train):
                │       ├─> read_index() — загрузка в RAM
                │       ├─> search(batch, k)
                │       └─> del index + gc.collect()
                └─> Merge кандидатов и top-k
    """
    PERSIST_POLITIC = StorageLevel.MEMORY_AND_DISK
    _SAMPLE_TARGET = 5_000_000
    # Лимит на то, сколько локальых индексов может быть одновременно загружено на драйвер
    DRIVER_INDEX_LIMIT = 5_000_000 
    PREDICT_SCHEMA = StructType([
        StructField("_id",         LongType(),  False),
        StructField("neighbor_id", LongType(),  False),
        StructField("distance",    FloatType(), False),
    ])
    CHUMK_SIZE = 512
    CLUSTERING_METHODS_MAPPER = {
        "k-means" : MiniBatchKMeans,
        "birch" : Birch
    }

    def __init__(
            self,
            # session: SparkSession,
            n_neighbors: int = 1,
            k: int = 1000,
            seed: Union[int, None] = None,
            feature_cols: Union[List[str], None] = None,
            faiss_mode: Literal["base", "partition", "auto"] = "auto",
            faiss_prefit_mode: Literal["sample", "full"] = "full",
            faiss_prefit_dict: dict[str, Any] = None
    ):
        """
        Args
        ----
            n_neighbors: `int`
                количество соседей для каждого запроса;

            k: `int`
                количество кластеров для пстроения FAISS IFV;

            seed: `Union[int, None]`
                зерно генерации для внутренних методов, по-умолчания `None`;

            feature_cols: `Union[List[str], None]`
                фичи по которым ищем соседей, по-умолчанию `None`;

            faiss_mode: `Literal["base", "partition", "auto"]`
                выбор алгоритма поиска соедей, по-умолчанию "auto".
                В "auto" моде выбирается режим в зависимости от размера выборки.

            faiss_prefit_mode: `Literal["sample", "full"]`
                выбор алгоритма формирования кластеров для faiss. 
                `sample` - обучается на сэмпле данных и использует внутренний k-means в faiss;
                `full` - батчами выгружает данные на драйвер, где обучает кластеризатор,
                центроиды кластеров которого далее используются в faiss.
            
            faiss_prefit_dict: `dict[str, Any]`
                словарь параметров для модели, долен содержать два поля: 
                `model` - название модели (`k-means`, `birch`),
                `params` - `dict` с параметрами модели.
        """
        self.n_neighbors = n_neighbors
        self.k = k
        self.seed = seed or 21
        self.feature_cols = feature_cols
        self.faiss_mode = faiss_mode
        self.faiss_prefit_mode = faiss_prefit_mode
        self.faiss_prefit_dict = faiss_prefit_dict

        self._index = None
        self._clustered_data: Optional[spark.DataFrame] = None
        self._mode: Optional[Literal["base", "partition"]] = None
        self._sharded_rdd: Optional[RDD] = None

    @staticmethod
    def _session_analizer(session: SparkSession) -> dict[str, Any]:
        """
        Анализатор спарк сессии.
        """        
        def _bytes_calculator(field: str) -> float:
            """
            Return
            ------
                `float` Размер в мб 
            """
            field = field.strip().lower()
            if field.endswith('m'):
                return float(field[:-1])
            if field.endswith('g'):
                return float(field[:-1]) * 1024
        
        defalt_config = {
            # "spark.python.worker.reuse" : True,
            # "spark.executor.memoryOverhead" : '',
            "spark.executor.memory" : "1g",
            "spark.executor.cores" : 1
        }
    
        sc = session.sparkContext
        executor_python_param = sc.getConf().get("spark.python.worker.reuse")
        overhead_memory = sc.getConf().get("spark.executor.memoryOverhead")
        executor_memory = sc.getConf().get("spark.executor.memory") or defalt_config["spark.executor.memory"]
        executor_cores = sc.getConf().get("spark.executor.cores") or defalt_config["spark.executor.cores"]


        executor_memory = _bytes_calculator(executor_memory)
        executor_cores = int(executor_cores)
        if executor_python_param is None or executor_python_param == "true":
            executor_python_param = True
        else: 
            executor_python_param = False
            raise SessionError("spark.python.worker.reuse")
        
        if overhead_memory is None:
            overhead_memory = 0.1 * executor_memory
        else:
            overhead_memory = _bytes_calculator(overhead_memory)
        
        return {
            "overhead_memory" : overhead_memory, 
            "executor_cores" : executor_cores
        }


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

    def _direct_fit(
            self, 
            data: spark.DataFrame
    ) -> None:
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

    def _partition_fit(
            self, 
            data: spark.DataFrame,
    ) -> None:
        """
        Реализация partition faiss fit. 
        Предполагается, что будет браться sample данных, 
        который отражает св-ва основной выборки.
        Для этого стоит, вообще говоря, проводить АА-тест.
        Здесь будем использовать обычный sample для удобства.

        Args
        ----
            data: `spark.DataFrame`
                Данные в которых мы ищем соседей.

            mode: `Literal["sample", "full"]`
                алгоритм обучение IVF-индекса. По-умолчанию `full`.
        Return
        ------
            None
        """
        import gc

        prepeared_data = self._vectorize_data(data)
        self._clustered_data = prepeared_data
        session = data.sparkSession

        if self.faiss_prefit_mode == "sample":
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
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self._index.train(X)
            print("=" * 70)
            print("Sample fit done.")
            print("=" * 70)
        elif self.faiss_prefit_mode == 'full':
            self._prefit(data=prepeared_data)
            print("=" * 70)
            print("Clusters prefit done.")
            print("=" * 70)
        else:
            raise ValueError(f"Incorrect prefit mode: {type(self.faiss_prefit_mode).__name__}")

        bc_index = session.sparkContext.broadcast(self._index)
        del self._index
        self._index = None
        gc.collect()

        features = ["_id", "features"]
        self._sharded_rdd = (
            prepeared_data
            .select(*features)
            .rdd
            .mapPartitions(lambda it: FaissSpark._partition_faiss(it, bc_index))
            .persist(self.PERSIST_POLITIC)
        )
        self._sharded_rdd.count()

    @staticmethod
    def _partition_faiss(
        iterator: Iterable, 
        bc_index: Broadcast
    ):
        """
        Fit на локально на каждой партиции на осонвании данных из sample-а.

        Args
        ----
            iterator: Iterable
                итератор внутри партиции. 
            
            bc_index: Broadcast
                заброадкащенный индекс.
        
        Return
        ------
            `Generator`: сериализованный индекс из каждой партиции.
        """
        import faiss
        import numpy as np

        index = bc_index.value
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

        # yield index_with_ids
        yield faiss.serialize_index(index_with_ids)

    def _prefit(
            self, 
            data: spark.DataFrame
        ) -> None:
        """
        Обучение IVF-индекса на всем data датасете путем итеративной выгрузки партиций на драйвер.
        
        Args
        ----
            data : `spark.DataFrame`
                Данные в которых мы ищем соседей.

        """
        if self.faiss_prefit_dict is None:
            raise ValueError("`faiss_prefit_dict` must be provided when faiss_prefit_mode='full'")
        
        model_name = self.faiss_prefit_dict["model"]
        model_params = self.faiss_prefit_dict["params"]

        if model_name == 'k-means' and "n_clusters" in model_params:
            if model_params["n_clusters"] == self.k:
                pass
            else:
                raise ValueError(f"K-means n_clusters and k should be equal. But k={self.k} and n_clusters={model_params['n_clusters']}")
        else:
            model_params["n_clusters"] = self.k

        model_cls = self.CLUSTERING_METHODS_MAPPER[model_name]
        model = model_cls(**model_params) #TODO: для k-means число кластеров должно быть = self.k
        
        batch_size = self.DRIVER_INDEX_LIMIT
        np_batch = None

        for batch in (
            data
            .select("features") #TODO: тут или ранее сделать обработку None
            .rdd
            .mapPartitions(lambda it: FaissSpark._partition_load(it, batch_size))
            .toLocalIterator()
        ):
            np_batch = np.array(batch, dtype=np.float32)
            model.partial_fit(np_batch)
            # self._index.train(np_batch)

        
        if np_batch is not None:
            del np_batch
            gc.collect()

        centroids = model.cluster_centers_ if model_name == 'k-means' else model.subcluster_centers_
        centroids = centroids.astype(np.float32)
        index_shape = centroids.shape[1] #TODO: тут или ранее сделать обработку None 
        # nlist = min(len(centroids), max(1, data.count() // 39)) 
        nlist = len(centroids)

        quantizer = faiss.IndexFlatL2(index_shape)
        quantizer.add(centroids)
        self._index = faiss.IndexIVFFlat(quantizer, index_shape, nlist)
        self._index.is_trained = True
        # self._index.train(centroids.astype(np.float32))

        self._clustering_model = model
        
    @staticmethod
    def _partition_load(partition_iter, batch_size):
        batch = []
        for row in partition_iter:
            batch.append(list(row["features"]))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _direct_predict(
            self, 
            test_data: spark.DataFrame
    ) -> List[List[Union[int, float]]]:
        """
        Нахождение индексов прямым методом faiss:
        собирает датасет и индексы на драйвере и ищет похоиъ.
        """
        session = test_data.sparkSession
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
    
        return session.createDataFrame(result, schema=FaissSpark.PREDICT_SCHEMA)

    def _partition_predict(
            self, 
            test_data: spark.DataFrame
    ) -> RDD[list]:
        """
        Реализация partition faiss predict.

        Steps
        -----
        1. Итеративно получаем сериализованные индексы с каждой партиции и
        записывем их в файл с расширегием `.index`, которое умеет обрабатывать faiss.

        2. Отправляем `.index` файлы на все партиции test-data, для которой ищем соседей
        в train-data.

        3. На каждой партиции итеративно загружаем в RAM строку test-data, 
        и для нее итеративно подгружаем файл с индексами, где ищем top_k = 
        n_neighbors, после чего удаляем индекс из оперативной памяти.

        4. Результаты обернуты в `Spark.Dataframe` с схемой 
        (_id `LongType`, neighbor_id `LongType`, distance `FloatType`).

        5. Удаление временных файлов и временной дирекятории после материализации
        результатов в датафрейм.

        Args
        ----
            test_data : `spark.DataFrame`
                Данные для которых мы ищем соседей.

        Return
        ------
            result: `RDD' результирующая таблица с индексами соседей и расстоянием до них
        """
        import gc

        session = test_data.sparkSession
        config_dict = self._session_analizer(session)
        tmp_dir = "__partition_indexes"
        # if os.
        os.makedirs(tmp_dir, exist_ok=True) 
        # os.mkdir(tmp_dir)
        index_files_list = []

        for partition_index, shard in enumerate(self._sharded_rdd.toLocalIterator()):
            partition_indexes = faiss.deserialize_index(shard)
            index_file_name = f"__{partition_index}_partition_index.index"
            faiss.write_index(
                partition_indexes,
                f"./{tmp_dir}/{index_file_name}" 
            )    
            session.sparkContext.addFile(f"./{tmp_dir}/{index_file_name}")
            index_files_list.append(index_file_name)

            del partition_indexes   # ← explicit release
            gc.collect()
        session.sparkContext.addPyFile("index_cacher.py")
        bc_index_files_list = session.sparkContext.broadcast(index_files_list)
        bc_n_neighbors = session.sparkContext.broadcast(self.n_neighbors)
        bc_chunk_size = session.sparkContext.broadcast(self.CHUMK_SIZE)
        bc_config_dict = session.sparkContext.broadcast(config_dict)

        result_rdd = test_data.rdd.mapPartitions(lambda it:
                                        FaissSpark._per_partition_predict(
                                        it, 
                                        bc_n_neighbors=bc_n_neighbors, 
                                        bc_index_files_list=bc_index_files_list,
                                        bc_chunk_size=bc_chunk_size,
                                        bc_config_dict=bc_config_dict
            )
        )

        result_df = (
            session.createDataFrame(result_rdd, schema=FaissSpark.PREDICT_SCHEMA)
            .persist(self.PERSIST_POLITIC)
        )
        result_df.count()
        
        # Удаляем все созданные промежуточные файлы
        tmp_files = os.listdir(tmp_dir)
        for file in tmp_files:
            os.remove(f"{tmp_dir}/{file}")
        os.rmdir(tmp_dir)

        return result_df 

    @staticmethod
    def  _per_partition_predict(
        shard_iter: Iterable,
        bc_n_neighbors: Broadcast,
        bc_index_files_list: Broadcast,
        bc_chunk_size: Broadcast,
        bc_config_dict: Broadcast
    ):
        """
        Predict локально на каждой партиции.

        Из загруженных `.index` файлов итеративно создаем для каждой строки в датафрейме
        новую колонку, где будут находится ближайшие соседи.

        Args
        ----
            shard_iter: `Iterable`
                Итератор по партиции.

            bc_n_neighbors: `Broadcast`
                Количество соседей для поиска.

            bc_index_files_list: `Broadcast`
                Список из номеров файлов, где хранятся построенные индексы каждой партиции.  

            bc_config_dict: `Broadcast`
                Словарь с конфигом сессии.
        """
        import faiss
        import numpy as np
        from pyspark import SparkFiles
        import gc
        import builtins
        from index_cacher import get_executor_cache

        # One shared cache across ALL partition tasks on this executor worker
        cache = get_executor_cache(bc_config_dict.value)
            
        real_n = bc_n_neighbors.value
        index_files = bc_index_files_list.value
        chunk_size = bc_chunk_size.value
        # if not hasattr(builtins, '_faiss_cache'):
        #     builtins._faiss_cache = CachingIndex(max_index=2)

        # cache = builtins._faiss_cache
   
        ## Реализация батчами с полной выгрузкой партиции
        def iter_chunk(it: Iterable, chunk_size: int):
            chunk = []
            amount = 0
            for row in it:
                chunk.append(row)
                amount += 1

                if amount >= chunk_size:
                    amount = 0
                    yield chunk
                    chunk =[]
                
            if chunk:
                yield chunk
        
        for chunk in iter_chunk(shard_iter, chunk_size):
            if not chunk:
                return
            query_ids = np.array([r["_id"] for r in chunk], dtype=np.int64)
            batch = np.array([list(r["features"]) for r in chunk], dtype=np.float32)  # (Q, d)
            del chunk
            gc.collect()        

            candidates = [[] for _ in range(len(query_ids))]
            for index_file in index_files:
                # tmp_index = faiss.read_index(SparkFiles.get(index_file))
                tmp_index = cache.get(index_file)
                tmp_index.nprobe = real_n
                k = min(real_n, tmp_index.ntotal)
                dists, nids = tmp_index.search(batch, k)   # (Q, k)
                del tmp_index
                gc.collect()

                for q_idx in range(len(query_ids)):
                    for rank in range(k):
                        nid = int(nids[q_idx, rank])
                        if nid >= 0:
                            candidates[q_idx].append((float(dists[q_idx, rank]), nid))

            for q_idx, qid in enumerate(query_ids):
                top = sorted(candidates[q_idx], key=lambda x: x[0])[:real_n]
                for dist, nid in top:
                    yield (int(qid), nid, dist)


        # # реализация итеративная, а не батчами
        # for element in shard_iter:
        #     top_n_list = []
        #     vec = list(element["features"])
        #     for index_file in index_files:
        #         tmp_index = faiss.read_index(SparkFiles.get(index_file))
        #         np_element = np.array(vec, dtype=np.float32).reshape(1, -1)
        #         dists, indexes = tmp_index.search(np_element, real_n)
        #         top_n_list.extend(list(zip(indexes[0], dists[0])))
        #         del tmp_index   # ← release before next load
        #         gc.collect()
        #     top_n_list = sorted(top_n_list, key=lambda x: x[1])[:real_n]
        #     yield top_n_list
            
            
    @staticmethod
    def _per_partition_batch_predict(
        shard_iter: Iterable, 
        bc_batch: Broadcast, 
        n_neighbors : Broadcast,
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
        import faiss
        import numpy as np
        batch = bc_batch.value

        for shard in shard_iter:
            index = faiss.deserialize_index(shard)
            if index.ntotal == 0:
                continue

            k = min(n_neighbors.value, index.ntotal)
            distances, ids = index.search(batch, k)

            for q_idx in range(len(batch)):
                for i in range(k):
                    if ids[q_idx][i] >= 0:
                        yield (q_idx, int(ids[q_idx][i]), float(distances[q_idx][i]))      

    def _calculation_mode(
            self, 
            count: int
    ) -> Literal["base", "partition"]:
        """
        Выбор типа вычисления faiss.

        Agrs:
        -----
            count: `int`
                Количество строка в датафрейме.
        
        Return
        ------
            `Literal["base", "partition"]`: вариант реализации алгоритма.
        """
        if self.faiss_mode in ("base", "partition"):
            return self.faiss_mode

        if count > 1_000_000:
            return "partition"
        return "base"
    
    def fit(
            self, 
            data: spark.DataFrame
    ) -> "FaissSpark":
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
        elif self._mode == "partition":
            self._partition_fit(data)

        return self
    
    def predict(
            self, 
            test_data: spark.DataFrame
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
            
        if self._sharded_rdd is not None:
            self._sharded_rdd.unpersist()
            self._sharded_rdd = None
    
    def __enter__(self) -> "FaissSpark":
        return self
        
    def __exit__(self, *_) -> None:
        self.unpersist()
            