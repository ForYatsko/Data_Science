import pandas as pd
import logging

def save_to_csv(data, file_path):
    """Сохраняет данные в CSV файл."""
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, list):
            pd.DataFrame(data).to_csv(file_path, index=False)
        logging.info(f"Данные успешно сохранены в файл: {file_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в CSV: {e}")

def save_to_json(data, file_path):
    """Сохраняет данные в JSON файл."""
    try:
        pd.DataFrame(data).to_json(file_path, orient="records", indent=4)
        logging.info(f"Данные успешно сохранены в файл: {file_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в JSON: {e}")