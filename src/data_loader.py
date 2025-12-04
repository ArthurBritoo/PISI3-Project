from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    @abstractmethod
    def load_data(self):
        pass

class CSVDataLoader(DataLoader):
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load_data(self):
        return pd.read_csv(self.filepath)

class ParquetDataLoader(DataLoader):
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load_data(self):
        return pd.read_parquet(self.filepath)