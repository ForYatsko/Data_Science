import os
import requests
import logging
import pandas as pd
from save import save_to_csv, save_to_json

# Импортируем настройки логгирования
from my_logging import setup_logging
setup_logging()

class DataLoader:
    def __init__(self):
        logging.info("DataLoader initialized.")

    def load_from_csv(self, file_path):
        if not os.path.isfile(file_path):
            logging.error(f"CSV file not found: {file_path}")
            return None

        logging.info(f"Loading data from CSV file: {file_path}")
        try:
            # Используем pandas для загрузки данных
            data = pd.read_csv(file_path)
            logging.info("Data successfully loaded from CSV.")
            return data
        except Exception as e:
            logging.error(f"Error loading data from CSV: {e}")
            return None
            
    def load_from_json(self, file_path):
        if not os.path.isfile(file_path):
            logging.error(f"JSON file not found: {file_path}")
            return None

        logging.info(f"Loading data from JSON file: {file_path}")
        try:
            data = pd.read_json(file_path)
            logging.info("Data successfully loaded from JSON.")
            return data
        except Exception as e:
            logging.error(f"Error loading data from JSON: {e}")
            return None

    def load_from_api(self, url):
        logging.info(f"Loading data from API: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            logging.info("Data successfully loaded from API.")
            return pd.json_normalize(data)  # Преобразование в DataFrame. DataFrame — двухмерная структура, состоящая из колонок и строк.
        except requests.RequestException as e:
            logging.error(f"Error loading data from API: {e}")
            return None

    def save_data(self, data, file_path):
        if file_path.endswith('.csv'):
            save_to_csv(data, file_path)
        elif file_path.endswith('.json'):
            save_to_json(data, file_path)
        else:
            logging.error("Unsupported file format for saving data.")