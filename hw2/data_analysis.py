import pandas as pd
import logging

# Настройка логгирования
from my_logging import setup_logging
setup_logging()

class DataAnalyzer:
    def __init__(self, data):
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        logging.info("DataAnalyzer initialized.")

    def count_missing_values(self):
        missing_counts = self.data.isnull().sum()
        logging.info("Missing values counted.")
        return missing_counts

    def report_missing_values(self):
        missing_info = self.count_missing_values()
        logging.info("Missing values report generated.")
        return missing_info[missing_info > 0]

    def fill_missing_values(self, strategy='mean'):
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                if strategy == 'mode':
                    # Для строковых данных (object) заполняем модой
                    self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
            else:
                # Для числовых данных
                if strategy == 'mean':
                    self.data[column] = self.data[column].fillna(self.data[column].mean())
                elif strategy == 'median':
                    self.data[column] = self.data[column].fillna(self.data[column].median())
                elif strategy == 'mode':
                    self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
                else:
                    logging.warning(f"Unknown strategy '{strategy}' for filling missing values.")
        logging.info("Missing values filled.")

    def drop_missing_values(self):
        original_count = self.data.shape[0]
        self.data.dropna(inplace=True)
        dropped_count = original_count - self.data.shape[0]
        logging.info(f"Dropped {dropped_count} rows with missing values.")

    def describe_data(self, percentiles=None):
        description = self.data.describe(percentiles=percentiles)
        logging.info("Data description generated.")
        return description

    def analyze_data_types(self):
        data_types = self.data.dtypes
        logging.info("Data types analyzed.")
        return data_types