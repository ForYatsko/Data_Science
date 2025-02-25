#This is data_loader.py
# Этот класс позволяет загружать данные из различных источников.
# Используем библиотеки pandas для работы с CSV и JSON, а также requests для обращения к API.
import pandas as pd
import requests

class DataLoader:  #Создаем класс DataLoader
    def __init__(self):  #Во всех классах есть специальный метод __init__() для инициализации его объектов. 
        pass  #pass оператор-бездельник, заглушка.

    def load_csv(self, file_path):  #В методах первым аргументом всегда идёт объект self. Он является объектом, для которого вызван метод.
        """Загрузка данных из CSV файла."""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            return None

    def load_json(self, file_path):
        """Загрузка данных из JSON файла."""
        try:
            data = pd.read_json(file_path)
            return data
        except Exception as e:
            print(f"Ошибка при загрузке JSON: {e}")
            return None

    def load_api(self, url):
        """Загрузка данных из API."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Проверка на ошибки
            data = response.json()
            return pd.json_normalize(data)  # Преобразование в DataFrame. DataFrame — двухмерная структура, состоящая из колонок и строк.
        except Exception as e:
            print(f"Ошибка при загрузке данных из API: {e}")
            return None