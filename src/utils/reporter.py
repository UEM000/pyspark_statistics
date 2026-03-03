from memory_profiler import memory_usage
from pyspark.sql import SparkSession
import time
import pandas as pd
from typing import Union


class Reporter():

    def __init__(self):
        self._start = None
        self._end = None
        self.mem_usage = []

    def start(self):
        self._start = time.time()
        self.mem_usage = []
        
    def stop(self):
        self._end = time.time()
    
    def memory_monitor(self, func, *args, **kwargs):
        mem_usage = memory_usage((func, args, kwargs), 
                                  interval=0.1, 
                                  timeout=None,
                                  max_usage=True)
        self.mem_usage.extend(mem_usage if isinstance(mem_usage, list) else [mem_usage])
        return func(*args, **kwargs)
    
    def get_report(self, deep: bool = False) -> Union[pd.DataFrame, dict, None]:
        if not self.mem_usage:
            return None

        if not deep:
            peak_memory_mb = max(self.mem_usage)
            avg_memory_mb = sum(self.mem_usage) / len(self.mem_usage)
            total_time = self._end - self._start if self._end else 0
            
            return {
                'peak_memory_Mb': peak_memory_mb,
                'avg_memory_Mb': avg_memory_mb,
                'execution_time': total_time,
            }
        
        df = pd.DataFrame({
            'timestamp': range(len(self.mem_usage)),
            'memory_mb': self.mem_usage
        })
        return df
    
class MemorySparkReporter(Reporter):

    def __init__(self, session: SparkSession):
        super().__init__()

        self.sparksession = session
        self.sc = self.sparksession.sparkContext
        self.stagemetrics = self.sc._jvm.ch.cern.sparkmeasure.StageMetrics(self.sparksession._jsparkSession)

    def start(self): 
        self.stagemetrics.begin()

    def stop(self):
        self.stagemetrics.end()

    def memory_monitor(self, func, *args, **kwargs):
        return func(*args, **kwargs)
    
    def get_report(self, deep = False):
        metrics_dict = self.stagemetrics.aggregateStageMetricsJavaMap()
        memory_usage = metrics_dict['peakExecutionMemory'] / (1024 * 1024)
        execution_time = metrics_dict['executorRunTime'] / 1000

        return {
            'peak_memory_Mb': memory_usage,
            # 'avg_memory_Mb': avg_memory_mb,
            'execution_time': execution_time,
        }