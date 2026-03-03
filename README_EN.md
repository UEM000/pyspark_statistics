# PySpark Statistics Benchmark

A framework for benchmarking and comparing the performance of statistical tests across different data processing engines: **PySpark**, **Pandas**, and **Polars**.

---

## 🌐 Language / Язык

[🇬🇧 English](README_EN.md) | [🇷🇺 Русский](README.md)

---

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API](#-api)
- [Architecture](#-architecture)
- [Usage Examples](#-usage-examples)

---

## 📖 About

This project is designed for conducting experiments to compare different implementations of statistical tests:

| Mode | Description |
|-------|----------|
| **Naive** | Classic implementation via scipy with data export |
| **Iterative** | Step-by-step calculation without data export (for 1 split) |
| **Parallel** | Vectorized calculation for all splits simultaneously |
| **Alternative** | Alternative optimized implementation (Spark only) |

### Supported Statistical Tests

| Test | Purpose | Status |
|------|------------|--------|
| **T-test** | Compare means of two samples | ✅ All implementations |
| **Chi-square** | Test homogeneity of categorical features | ✅ All implementations |
| **KS-test** | Test equality of distributions | ⚠️ Only naive (scipy) |

---

## ✨ Features

- 🚀 **Multi-engine support** — unified interface for Spark, Pandas, and Polars
- 📊 **Flexible configuration** — experiment setup via config.py
- 📈 **Resource monitoring** — memory and execution time tracking
- 🔄 **K-split validation** — support for multiple data splits
- 🎯 **Automatic grid search** — parameter combinations exploration

---

## 📦 Installation

### Requirements

- Python 3.8+
- PySpark 3.x
- Pandas 1.x
- Polars 0.18+
- SciPy 1.x

### Install Dependencies

```bash
pip install pyspark pandas polars scipy numpy memory-profiler tqdm
```

### Optional (for Spark metrics)

```bash
# spark-measure for detailed Spark metrics
spark.jars.packages: ch.cern.sparkmeasure:spark-measure_2.12:0.23
```

---

## 🗂 Project Structure

```
pyspark_statistics/
├── src/
│   ├── experiment/              # Experiment module
│   │   ├── __init__.py
│   │   ├── config.py            # Test configuration
│   │   └── experiment.py        # Experiment class
│   │
│   ├── statistic_realization/   # Test implementations
│   │   ├── __init__.py
│   │   ├── spark_statistic.py   # Spark implementations
│   │   ├── pandas_statistic.py  # Pandas implementations
│   │   └── polars_statistic.py  # Polars implementations
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── enums.py             # Data type enums
│       ├── spliter.py           # Group splitting
│       └── reporter.py          # Resource monitoring
│
├── datasets/                    # Test data
│   ├── data.csv
│   └── data_no_index.csv
│
├── tests/                       # Tests
│   ├── session.ipynb
│   └── testing pandas.ipynb
│
├── benchmark_spark.ipynb        # Spark benchmark
├── benchmark_pandas.ipynb       # Pandas benchmark
├── benchmark_polars.ipynb       # Polars benchmark
└── README.md
```

---

## 🚀 Quick Start

### Spark Example

```python
from src.experiment import Experiment
from pyspark.sql import SparkSession

# Initialize Spark session
session = (
    SparkSession.builder
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()
)

# Load data
df = (
    session.read.format('csv')
    .option("header", "true")
    .option("inferSchema", "true")
    .load("./datasets/data.csv")
)

# Create and run experiment
experiment = Experiment(
    data=df,
    result_path="./results/",
    session=session,
    report=True,           # Save memory/time reports
    output=True,           # Save test results
    output_path="./output.txt"
)

experiment.execute()
```

### Pandas Example

```python
import pandas as pd
from src.experiment import Experiment
from pyspark.sql import SparkSession

# Data
df = pd.read_csv("./datasets/data.csv")

# Empty session (required for compatibility)
session = SparkSession.builder.getOrCreate()

experiment = Experiment(
    data=df,
    result_path="./results_pandas/",
    session=session,
    report=True
)

experiment.execute()
```

### Polars Example

```python
import polars as pl
from src.experiment import Experiment
from pyspark.sql import SparkSession

# Data (DataFrame or LazyFrame)
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

## ⚙️ Configuration

### File `src/experiment/config.py`

#### Experiment List

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

| Parameter | Description | Possible Values |
|----------|----------|-----------------|
| `test_type` | Statistical test types | `ttest`, `chisquare`, `kstest` |
| `algo_type` | Implementation modes | `naive`, `iterative`, `parallel`, `alternative` |

#### Experiment Parameters

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

| Parameter | Description |
|----------|----------|
| `random_state` | Seed for reproducibility |
| `groups_num` | Number of groups for splitting |
| `target_category_columns` | Categorical columns for Chi-square |
| `target_numeric_columns` | Numeric columns for T-test/KS-test |
| `k_splits` | Number of splits |
| `fractions` | Data fraction for sampling |
| `target_frac` | Feature fraction for testing |

---

## 📚 API

### `Experiment` Class

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

| Parameter | Description |
|----------|----------|
| `data` | Source data (Spark/Pandas/Polars) |
| `result_path` | Path for saving reports |
| `session` | Spark session (required) |
| `report` | Enable memory/time monitoring |
| `output` | Save test results |
| `output_path` | Path for test results |

### Methods

| Method | Description |
|-------|----------|
| `execute()` | Run all experiments from `test_list` |

### Test Classes

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

### Utilities

#### Spliter

```python
from src.utils import StandartSpliter, BinarySpliter

# Single split
spliter = StandartSpliter(data, groups_num=4)
df_with_groups = spliter.split()

# Multiple splits (K-split)
spliter = BinarySpliter(data, groups_num=4, k_splits=5)
df_with_groups = spliter.split()
```

#### Reporter

```python
from src.utils import Reporter

reporter = Reporter()
reporter.start()

# Tracked function
result = reporter.memory_monitor(my_function, arg1, arg2)

reporter.stop()
report = reporter.get_report()
# {'peak_memory_Mb': ..., 'avg_memory_Mb': ..., 'execution_time': ...}
```

---

## 🏗 Architecture

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

## 📝 Usage Examples

### Setting Up a Custom Experiment

1. Open `src/experiment/config.py`

2. Add a new experiment:

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

3. Configure parameters:

```python
experiment_data["variative var"] = {
    "k_splits": [2, 4, 8],
    "fractions": [0.1, 0.5, 1.0],
    "target_frac": [0.5, 1.0]
}
```

4. Run:

```python
experiment.execute()
```

### Interpreting Results

Output file format:

```
test_label=ttest; realization=1; column=age; permutation=split: 0 groups: 0, 1; p-value=0.03421; statistic=2.145; pass=True;
```

| Field | Description |
|------|----------|
| `test_label` | Test name |
| `realization` | Mode (0=naive, 1=iterative, 2=parallel, 3=alternative) |
| `column` | Tested column |
| `permutation` | Split and group pairs |
| `p-value` | Test p-value |
| `statistic` | Test statistic |
| `pass` | `True` if significant (p < 0.05) |

---

## 📊 Data Format

Expected input dataset structure:

| Column | Type | Description |
|--------|------|-------------|
| `industry` | category | Category for Chi-square |
| `gender` | category | Category for Chi-square |
| `pre_spends` | numeric | Numeric feature for T-test |
| `post_spends` | numeric | Numeric feature for T-test |
| `age` | numeric | Numeric feature for T-test/KS-test |
| `group` | auto | Generated automatically |


