import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import numpy as np


class DataVisualizer:
    def __init__(self, data: pd.DataFrame = None):
        """
        Инициализация визуализатора данных.

        :param data: pandas DataFrame с данными для визуализации (необязательно).
        """
        if data is not None and not isinstance(data, pd.DataFrame):
            raise ValueError("Ожидается объект pandas DataFrame или None.")
        self.data = data

    def plot_histogram(self, column: str, title: str, bins: int = 10, color: str = "blue"):
        """
        Строит гистограмму для указанного столбца.

        :param column: Столбец для построения гистограммы.
        :param title: Заголовок графика.
        :param bins: Количество бинов.
        :param color: Цвет гистограммы.
        """
        if self.data is None:
            logging.error("DataFrame не был передан в визуализатор.")
            return

        if column not in self.data.columns:
            logging.error(f"Столбец '{column}' не найден в данных.")
            return

        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=column, bins=bins, color=color, kde=True)
            plt.title(title)
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
            logging.info(f"Гистограмма для столбца '{column}' успешно построена.")
        except Exception as e:
            logging.error(f"Ошибка при построении гистограммы для столбца '{column}': {e}")

    def plot_scatter(self, x_column: str, y_column: str, title: str, color: str = "red"):
        """
        Строит диаграмму рассеяния для двух столбцов.

        :param x_column: Столбец для оси X.
        :param y_column: Столбец для оси Y.
        :param title: Заголовок графика.
        :param color: Цвет точек.
        """
        if self.data is None:
            logging.error("DataFrame не был передан в визуализатор.")
            return

        if x_column not in self.data.columns or y_column not in self.data.columns:
            missing_columns = [col for col in [x_column, y_column] if col not in self.data.columns]
            logging.error(f"Столбцы {missing_columns} не найдены в данных.")
            return

        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.data, x=x_column, y=y_column, color=color)
            plt.title(title)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.tight_layout()
            plt.show()
            logging.info(f"Диаграмма рассеяния для столбцов '{x_column}' и '{y_column}' успешно построена.")
        except Exception as e:
            logging.error(f"Ошибка при построении диаграммы рассеяния: {e}")

    def plot_combined_metrics(self, top_classifiers: pd.DataFrame, metrics: list, title: str, filename: str = None):
        """
        Строит группированный график метрик для топ-5 классификаторов.

        :param top_classifiers: DataFrame с топ-5 классификаторов. Ожидается наличие столбцов метрик.
        :param metrics: Список метрик для отображения.
        :param title: Заголовок графика.
        :param filename: Путь для сохранения графика (опционально).
        """
        try:
            # Убедимся, что метрики есть в датафрейме
            for metric in metrics:
                if metric not in top_classifiers.columns:
                    logging.error(f"Метрика '{metric}' отсутствует в DataFrame.")
                    return

            # Настройка размеров графика
            plt.figure(figsize=(12, 6))

            # Количество классификаторов и метрик
            num_classifiers = top_classifiers.shape[0]
            num_metrics = len(metrics)

            # Создаём группированный график
            x = np.arange(num_classifiers)  # Позиции для классификаторов
            width = 0.2  # Ширина столбца для каждой метрики

            for i, metric in enumerate(metrics):
                plt.bar(
                    x + i * width,
                    top_classifiers[metric],
                    width=width,
                    label=metric
                )

            # Настраиваем оси и подписи
            plt.xticks(x + width * (num_metrics - 1) / 2, top_classifiers["Classifier"], rotation=45, fontsize=12)
            plt.xlabel("Классификатор", fontsize=14)
            plt.ylabel("Значение метрик", fontsize=14)
            plt.title(title, fontsize=16)
            plt.legend(title="Метрики", fontsize=12)
            plt.tight_layout()

            # Сохранение графика (если указан путь)
            if filename:
                plt.savefig(filename)
                logging.info(f"График метрик топ-5 классификаторов сохранен в файл: {filename}")

            # Отображение графика
            plt.show()
        except Exception as e:
            logging.error(f"Ошибка при построении графика метрик: {e}")

    def plot_correlation_matrix(self, title: str = "Корреляционная матрица", cmap: str = "coolwarm"):
        """
        Строит тепловую карту корреляционной матрицы для числовых данных.

        :param title: Заголовок графика.
        :param cmap: Цветовая карта для тепловой карты.
        """
        if self.data is None:
            logging.error("DataFrame не был передан в визуализатор.")
            return

        try:
            # Вычисляем корреляционную матрицу
            corr_matrix = self.data.corr()

            if corr_matrix.empty:
                logging.error("Корреляционная матрица пуста. Убедитесь, что в данных есть числовые признаки.")
                return

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5)
            plt.title(title)
            plt.tight_layout()
            plt.show()
            logging.info("Корреляционная матрица успешно визуализирована.")
        except Exception as e:
            logging.error(f"Ошибка при визуализации корреляционной матрицы: {e}")

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, title: str, filename: str, show: bool = True):
        """
        Строит тепловую диаграмму для таблицы корреляции и сохраняет её в файл.

        :param correlation_matrix: Матрица корреляции (DataFrame).
        :param title: Заголовок для тепловой диаграммы.
        :param filename: Путь для сохранения изображения.
        :param show: Если True, отображает график после сохранения.
        """
        try:
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".1f",
                cmap="coolwarm",
                square=True,
                cbar=True,
                annot_kws={"size": 10},
                linewidths=0.5
            )
            plt.title(title, fontsize=18)
            plt.xticks(fontsize=12, rotation=45, ha="right")
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(filename)
            logging.info(f"Тепловая диаграмма корреляции сохранена в файл: {filename}")
            if show:
                plt.show()
            plt.close()
        except Exception as e:
            logging.error(f"Ошибка при построении тепловой диаграммы: {e}")

    def plot_class_distribution(self, y: pd.Series, class_names: dict, output_folder: str):
        """
        Построение круговой диаграммы распределения классов.

        :param y: Целевая переменная (Series).
        :param class_names: Словарь с названиями классов.
        :param output_folder: Папка для сохранения графика.
        """
        try:
            # Получаем количество экземпляров каждого класса
            class_counts = y.value_counts().rename(index=class_names)

            # Построение круговой диаграммы
            plt.figure(figsize=(8, 8))
            plt.pie(
                class_counts,
                labels=class_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
            )
            plt.title("Распределение классов в целевой переменной", fontsize=16)
            plt.tight_layout()

            # Сохранение графика в указанный файл
            save_path = f"{output_folder}/class_distribution.png"
            plt.savefig(save_path)
            plt.show()
            logging.info(f"График распределения классов сохранен: {save_path}")
        except Exception as e:
            logging.error(f"Ошибка при построении графика распределения классов: {e}")

    def plot_classifier_metrics(self, results: pd.DataFrame, metric: str, title: str):
        """
        Построение графика классификаторов: названия по оси X, метрики по оси Y.

        :param results: DataFrame с результатами классификации. Ожидается наличие колонок:
                        - "Classifier" (названия классификаторов)
                        - метрика (например, "Accuracy", "F1 Score", и т.д.)
        :param metric: Название метрики для отображения на оси Y.
        :param title: Заголовок графика.
        """
        if metric not in results.columns or "Classifier" not in results.columns:
            logging.error(f"Метрика '{metric}' или столбец 'Classifier' отсутствуют в DataFrame.")
            return

        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(data=results, x="Classifier", y=metric, palette="viridis")
            plt.title(title, fontsize=16)
            plt.xlabel("Классификатор", fontsize=14)
            plt.ylabel(metric, fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.tight_layout()
            plt.show()
            logging.info(f"График для метрики '{metric}' успешно построен.")
        except Exception as e:
            logging.error(f"Ошибка при построении графика для метрики '{metric}': {e}")