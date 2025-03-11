import csv
import json
import logging

# Импортируем настройки логгирования
from my_logging import setup_logging

setup_logging()  # Настройка логгирования

def save_to_csv(data, file_path):
    """
    Сохраняет данные в CSV файл.

    :param data: Список словарей, представляющих строки данных.
    :param file_path: Путь к CSV файлу.
    """
    if not data:
        logging.warning("Нет данных для сохранения в CSV.")
        return

    logging.info(f"Сохранение данных в CSV файл: {file_path}")
    try:
        with open(file_path, mode='w', encoding='utf-8', newline='') as csvfile:
            # Проверяем, что data - список словарей
            if isinstance(data, list) and isinstance(data[0], dict):
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                logging.info(f"Данные успешно сохранены в CSV. Количество строк: {len(data)}")
            else:
                raise ValueError("Некорректный формат данных для CSV. Ожидается список словарей.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в CSV: {e}")
        raise  # Повторно выбрасываем исключение, чтобы основной код мог обработать его

def save_to_json(data, file_path):
    """
    Сохраняет данные в JSON файл.

    :param data: Данные для сохранения (любой сериализуемый объект).
    :param file_path: Путь к JSON файлу.
    """
    if data is None:
        logging.warning("Нет данных для сохранения в JSON.")
        return

    logging.info(f"Сохранение данных в JSON файл: {file_path}")
    try:
        with open(file_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, ensure_ascii=False, indent=4)
            logging.info("Данные успешно сохранены в JSON.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных в JSON: {e}")
        raise  # Повторно выбрасываем исключение, чтобы основной код мог обработать его