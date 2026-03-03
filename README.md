# PySpark Statistics Benchmark

Фреймворк для бенчмаркинга и сравнения производительности статистических тестов на различных движках обработки данных: **PySpark**, **Pandas** и **Polars**.

---

## 🌐 Language / Язык

[🇬🇧 English](README_EN.md) | [🇷🇺 Русский](README.md)

---

## 📋 Содержание

- [О проекте](#-о-проекте)
- [Возможности](#-возможности)
- [Установка](#-установка)
- [Структура проекта](#-структура-проекта)
- [Быстрый старт](#-быстрый-старт)
- [Конфигурация](#-конфигурация)
- [API](#-api)
- [Архитектура](#-архитектура)
- [Примеры использования](#-примеры-использования)

---

## 📖 О проекте

Этот проект предназначен для проведения экспериментов по сравнению различных реализаций статистических тестов:

| Режим | Описание |
|-------|----------|
| **Naive** | Классическая реализация через scipy с выгрузкой данных |
| **Iterative** | Пошаговый расчёт без выгрузки данных (для 1 сплита) |
| **Parallel** | Векторизованный расчёт для всех сплитов одновременно |
| **Alternative** | Альтернативная оптимизированная реализация (Spark) |

### Поддерживаемые статистические тесты

| Тест | Назначение | Статус |
|------|------------|--------|
| **T-test** | Сравнение средних двух выборок | ✅ Все реализации |
| **Chi-square** | Проверка однородности категориальных признаков | ✅ Все реализации |
| **KS-test** | Проверка равенства распределений | ⚠️ Только naive (scipy) |

---

## ✨ Возможности

- 🚀 **Мультидвижковая поддержка** — единый интерфейс для Spark, Pandas и Polars
- 📊 **Гибкая конфигурация** — настройка параметров эксперимента через config.py
- 📈 **Мониторинг ресурсов** — замер памяти и времени выполнения
- 🔄 **K-split валидация** — поддержка множественных разбиений данных
- 🎯 **Автоматический grid search** — перебор комбинаций параметров

---

## 📦 Установка

### Требования

- Python 3.8+
- PySpark 3.x
- Pandas 1.x
- Polars 0.18+
- SciPy 1.x

### Установка зависимостей

```bash
pip install pyspark pandas polars scipy numpy memory-profiler tqdm
```

### Опционально (для Spark-метрик)

```bash
# spark-measure для детальной метрики Spark
spark.jars.packages: ch.cern.sparkmeasure:spark-measure_2.12:0.23
```

---

## 🗂 Структура проекта

```
pyspark_statistics/
├── src/
│   ├── experiment/              # Модуль экспериментов
│   │   ├── __init__.py
│   │   ├── config.py            # Конфигурация тестов
│   │   └── experiment.py        # Класс Experiment
│   │
│   ├── statistic_realization/   # Реализации тестов
│   │   ├── __init__.py
│   │   ├── spark_statistic.py   # Spark-реализации
│   │   ├── pandas_statistic.py  # Pandas-реализации
│   │   └── polars_statistic.py  # Polars-реализации
│   │
│   └── utils/                   # Утилиты
│       ├── __init__.py
│       ├── enums.py             # Enum типов данных
│       ├── spliter.py           # Разбиение на группы
│       └── reporter.py          # Мониторинг ресурсов
│
├── datasets/                    # Тестовые данные
│   ├── data.csv
│   └── data_no_index.csv
│
├── tests/                       # Тесты
│   ├── session.ipynb
│   └── testing pandas.ipynb
│
├── benchmark_spark.ipynb        # Бенчмарк Spark
├── benchmark_pandas.ipynb       # Бенчмарк Pandas
├── benchmark_polars.ipynb       # Бенчмарк Polars
└── README.md
```

---

## 🚀 Быстрый старт

### Пример запуска на Spark

```python
from src.experiment import Experiment
from pyspark.sql import SparkSession

# Инициализация Spark сессии
session = (
    SparkSession.builder
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

# Загрузка данных
df = (
    session.read.format('csv')
    .option("header", "true")
    .option("inferSchema", "true")
    .load("./datasets/data.csv")
)

# Создание и запуск эксперимента
experiment = Experiment(
    data=df,
    result_path="./results/",
    session=session,
    report=True,           # Сохранять отчёты о памяти/времени
    output=True,           # Сохранять результаты тестов
    output_path="./output.txt"
)

experiment.execute()
```

### Пример на Pandas

```python
import pandas as pd
from src.experiment import Experiment
from pyspark.sql import SparkSession

# Данные
df = pd.read_csv("./datasets/data.csv")

# Пустая сессия (требуется для совместимости)
session = SparkSession.builder.getOrCreate()

experiment = Experiment(
    data=df,
    result_path="./results_pandas/",
    session=session,
    report=True
)

experiment.execute()
```

### Пример на Polars

```python
import polars as pl
from src.experiment import Experiment
from pyspark.sql import SparkSession

# Данные (DataFrame или LazyFrame)
df = pl.read_csv("./datasets/data.csv").lazy()

session = SparkSession.builder.getOrCreate()

experiment = Experiment(
    data=df,
    result_path="./results_polars/",
    session=session,
    report=True
)

experiment.execute()
```

---

## ⚙️ Конфигурация

### Файл `src/experiment/config.py`

#### Список экспериментов

```python
test_list = [
    {
        "all_tests_01": {
            "test_type": ["chisquare", "ttest", "kstest"],
            "algo_type": ["naive", "iterative"]
        }
    }
]
```

| Параметр | Описание | Возможные значения |
|----------|----------|-------------------|
| `test_type` | Типы статистических тестов | `ttest`, `chisquare`, `kstest` |
| `algo_type` | Режимы реализации | `naive`, `iterative`, `parallel`, `alternative` |

#### Параметры эксперимента

```python
experiment_data = {
    "constant var": {
        "random_state": 21,
        "groups_num": 4,
        "target_category_columns": ["industry", "gender"],
        "target_numeric_columns": ["pre_spends", "post_spends", "age"]
    },
    "variative var": {
        "k_splits": [2],
        "fractions": [1],
        "target_frac": [1]
    }
}
```

| Параметр | Описание |
|----------|----------|
| `random_state` | Seed для воспроизводимости |
| `groups_num` | Количество групп для разбиения |
| `target_category_columns` | Категориальные колонки для Chi-square |
| `target_numeric_columns` | Числовые колонки для T-test/KS-test |
| `k_splits` | Количество разбиений (сплитов) |
| `fractions` | Доля данных для сэмплирования |
| `target_frac` | Доля признаков для тестирования |

---

## 📚 API

### Класс `Experiment`

```python
Experiment(
    data: Union[DataFrame, pd.DataFrame, pl.DataFrame],
    result_path: str,
    session: SparkSession,
    report: bool = False,
    output: bool = False,
    output_path: str = None
)
```

| Параметр | Описание |
|----------|----------|
| `data` | Исходные данные (Spark/Pandas/Polars) |
| `result_path` | Путь для сохранения отчётов |
| `session` | Spark сессия (обязательна) |
| `report` | Включить мониторинг памяти/времени |
| `output` | Сохранять результаты тестов |
| `output_path` | Путь для результатов тестов |

### Методы

| Метод | Описание |
|-------|----------|
| `execute()` | Запуск всех экспериментов из `test_list` |

### Классы тестов

#### Spark

```python
from src.statistic_realization import TestingTTest, TestingChiSquare, TestingKStest
```

#### Pandas

```python
from src.statistic_realization import TestingTTestPandas, TestingChiSquarePandas, TestingKStestPandas
```

#### Polars

```python
from src.statistic_realization import Ttest, ChiSquare, KStest
```

### Утилиты

#### Spliter

```python
from src.utils import StandartSpliter, BinarySpliter

# Однократное разбиение
spliter = StandartSpliter(data, groups_num=4)
df_with_groups = spliter.split()

# Множественное разбиение (K-split)
spliter = BinarySpliter(data, groups_num=4, k_splits=5)
df_with_groups = spliter.split()
```

#### Reporter

```python
from src.utils import Reporter

reporter = Reporter()
reporter.start()

# Отслеживаемая функция
result = reporter.memory_monitor(my_function, arg1, arg2)

reporter.stop()
report = reporter.get_report()
# {'peak_memory_Mb': ..., 'avg_memory_Mb': ..., 'execution_time': ...}
```

---

## 🏗 Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                     Experiment                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ config.py   │  │ test_list   │  │ permutations grid   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│     Spark       │ │     Pandas      │ │     Polars      │
│  TestingTTest   │ │ TestingTTest    │ │     Ttest       │
│ TestingChiSquare│ │ TestingChiSquare│ │    ChiSquare    │
│  TestingKStest  │ │ TestingKStest   │ │     KStest      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │      Utils          │
                   │  Spliter / Reporter │
                   └─────────────────────┘
```

---

## 📝 Примеры использования

### Настройка собственного эксперимента

1. Откройте `src/experiment/config.py`

2. Добавьте новый эксперимент:

```python
test_list = [
    {
        "my_custom_experiment": {
            "test_type": ["ttest"],
            "algo_type": ["naive", "parallel"]
        }
    }
]
```

3. Настройте параметры:

```python
experiment_data["variative var"] = {
    "k_splits": [2, 4, 8],
    "fractions": [0.1, 0.5, 1.0],
    "target_frac": [0.5, 1.0]
}
```

4. Запустите:

```python
experiment.execute()
```

### Интерпретация результатов

Формат выходного файла:

```
test_label=ttest; realization=1; column=age; permutation=split: 0 groups: 0, 1; p-value=0.03421; statistic=2.145; pass=True;
```

| Поле | Описание |
|------|----------|
| `test_label` | Название теста |
| `realization` | Режим (0=naive, 1=iterative, 2=parallel, 3=alternative) |
| `column` | Тестируемая колонка |
| `permutation` | Сплит и пары групп |
| `p-value` | P-значение теста |
| `statistic` | Статистика теста |
| `pass` | `True` если значимо (p < 0.05) |

---

## 📊 Формат данных

Ожидаемая структура входного датасета:

| Column | Type | Description |
|--------|------|-------------|
| `industry` | category | Категория для Chi-square |
| `gender` | category | Категория для Chi-square |
| `pre_spends` | numeric | Числовой признак для T-test |
| `post_spends` | numeric | Числовой признак для T-test |
| `age` | numeric | Числовой признак для T-test/KS-test |
| `group` | auto | Генерируется автоматически |

