from collections import OrderedDict
from pyspark import SparkFiles
import faiss
import builtins
import threading

class CachingIndex:
    """
    Класс для кэширования индекса на экзекъюторе.
    Предназначен для того, чтобы в памяти экзекъютора, при одновременном 
    выполнении нескольких партиций не происходило ситуации, один и тот же индекс
    train-data-ы не материализовывался несколько раз
    """

    def __init__(
        self,
        max_index: int=2
    ):
        """
        Args
        ----
            max_index : `int`
                Колчиество индексов в памяти одновременно.
        """
        self._max = max_index
        self._cache = OrderedDict()
        self._lock = threading.Lock()
    
    def get(
            self,
            index_file: int
    ):
        """
        Получаем индексы по заданному названию файла. Если такой файл уже обрабатывался,
        то просто выгружаем его из словаря и двигаем в последовательности ключей в конец.
        если такого файла нет в нашем кэше, то очищаем первый элемент и записываем в конец
        новый индекс.

        Args
        ----
            index_file : `str`
                Путь до файла с индексами, который выступает ключем.

        Return
        ------
            Возвращает FAISS индексы для заданного файла.
        """
        with self._lock:
            if index_file in self._cache:
                self._cache.move_to_end(index_file=index_file)
                return self._cache[index_file]

            if len(self._cache) == self._max:
                _, evicted = self._cache.popitem(last=False)
                del evicted
                import gc; gc.collect()
            
            tmp_index = faiss.read_index(SparkFiles.get(index_file))
            self._cache[index_file] = tmp_index
            return tmp_index
    
def get_executor_cache(max_index: int = 2) -> CachingIndex:
    if not hasattr(builtins, '_faiss_index_cache'):
        builtins._faiss_index_cache = CachingIndex(max_index=max_index)
    return builtins._faiss_index_cache