import enum

@enum.unique
class DataRealization(enum.Enum):
    spark = 0
    pandas = 1