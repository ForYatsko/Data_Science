import pandas as pd
import logging

class DataLoader:
    def load_from_csv(self, file_path):
        """Загружает данные из CSV файла."""
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Данные успешно загружены из {file_path}")
            return data
        except FileNotFoundError:
            logging.error(f"Файл не найден: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Ошибка при загрузке данных из CSV: {e}")
            return None