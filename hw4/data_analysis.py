import numpy as np  # Добавляем импорт numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalyzer:
    def __init__(self, df):
        """Инициализация анализатора данных."""
        self.df = df

    def analyze_data(self):
        """Анализ структуры данных и основных статистик."""
        print("Информация о датасете:")
        print(self.df.info())
        print("\nОписание данных:")
        print(self.df.describe())

    def check_missing_values(self):
        """Проверка на пропущенные значения."""
        print("\nПроверка на пропущенные значения:")
        missing_values = self.df.isnull().sum()
        print(missing_values)
        return missing_values

    def fill_missing_values(self):
        """Заполнение пропущенных значений медианой."""
        if self.df.isnull().sum().sum() > 0:  # Если есть пропущенные значения
            self.df.fillna(self.df.median(), inplace=True)
            print("\nПропущенные значения заполнены медианой.")
        else:
            print("\nПропущенные значения отсутствуют.")

    def plot_correlation_matrix(self):
        """Построение корреляционной матрицы."""
        print("\nПостроение корреляционной матрицы...")
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.corr()  # Расчёт корреляции
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Корреляционная матрица признаков")
        plt.show()

    def remove_highly_correlated_features(self, threshold=0.9):
        """
        Удаляет сильно коррелирующие признаки из DataFrame.

        :param threshold: Порог корреляции для удаления (по умолчанию 0.9).
        :return: Список удалённых признаков.
        """
        print("\nУдаление сильно коррелирующих признаков...")
        corr_matrix = self.df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)  # Используем numpy вместо pd.np
        )
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.df.drop(columns=to_drop, inplace=True)
        print(f"Удалено {len(to_drop)} признаков: {to_drop}")
        return to_drop

    def get_dataframe(self):
        """Возвращает обработанный DataFrame."""
        return self.df

    def check_class_balance(self, target_column, class_names):
        """
        Проверка сбалансированности данных.

        :param target_column: Название столбца с целевой переменной.
        :param class_names: Словарь с названиями классов.
        :return: None
        """
        y = self.df[target_column]
        class_counts = y.value_counts()
        total_samples = len(y)
        class_ratios = class_counts / total_samples

        print("\nОценка сбалансированности данных:")
        for class_id, count in class_counts.items():
            class_name = class_names.get(class_id, f"Класс {class_id}")
            ratio = class_ratios[class_id]
            print(f"Класс {class_name}: {count} образцов ({ratio:.2%})")

        # Оценка баланса
        min_ratio = class_ratios.min()
        max_ratio = class_ratios.max()

        if max_ratio / min_ratio > 1.5:
            print("\nВывод: Данные несбалансированы.")
        else:
            print("\nВывод: Данные сбалансированы.")