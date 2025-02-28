import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

# Настройка логгирования
from my_logging import setup_logging
setup_logging()

class DataVisualizer:
    def __init__(self, data):
        self.data = data
        logging.info("DataVisualizer initialized.")

    def _ensure_directory(self, path):
        """Убедитесь, что папка для сохранения существует."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Создана папка для сохранения визуализаций: {path}")

    def _generate_filename(self, prefix, save_path, extension='jpg'):
        """Генерирует уникальное имя файла для сохранения графика."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(save_path, f"{prefix}_{timestamp}.{extension}")

    def plot_histogram(self, column, color='blue', title='Histogram', xlabel='Values', ylabel='Frequency', save_path='visualizations'):
        """Построение гистограммы."""
        try:
            self._ensure_directory(save_path)
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column], bins=30, color=color)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            file_name = self._generate_filename('histogram', save_path)
            plt.savefig(file_name)
            logging.info(f"Гистограмма сохранена в файл: {file_name}")
            plt.show()
        except KeyError:
            logging.error(f"Столбец '{column}' не найден в данных.")
        except Exception as e:
            logging.error(f"Ошибка при построении гистограммы: {e}")

    def plot_line(self, x_column, y_column, color='green', title='Line Plot', xlabel='X-axis', ylabel='Y-axis', save_path='visualizations'):
        """Построение линейного графика."""
        try:
            self._ensure_directory(save_path)
            plt.figure(figsize=(10, 6))
            plt.plot(self.data[x_column], self.data[y_column], color=color)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            file_name = self._generate_filename('line', save_path)
            plt.savefig(file_name)
            logging.info(f"Линейный график сохранен в файл: {file_name}")
            plt.show()
        except KeyError as e:
            logging.error(f"Столбец '{e.args[0]}' не найден в данных.")
        except Exception as e:
            logging.error(f"Ошибка при построении линейного графика: {e}")

    def plot_scatter(self, x_column, y_column, color='red', title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis', save_path='visualizations'):
        """Построение диаграммы рассеяния."""
        try:
            self._ensure_directory(save_path)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.data[x_column], y=self.data[y_column], color=color)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            
            file_name = self._generate_filename('scatter', save_path)
            plt.savefig(file_name)
            logging.info(f"Диаграмма рассеяния сохранена в файл: {file_name}")
            plt.show()
        except KeyError as e:
            logging.error(f"Столбец '{e.args[0]}' не найден в данных.")
        except Exception as e:
            logging.error(f"Ошибка при построении диаграммы рассеяния: {e}")