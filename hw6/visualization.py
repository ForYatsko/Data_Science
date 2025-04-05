import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np

def visualize_statistics(df):
    """
    Визуализация распределения данных с гистограммами и описательной статистикой.
    """
    logging.info("Визуализация распределения данных с гистограммами...")
    try:
        # Список столбцов для визуализации
        columns_to_plot = ["Open", "High", "Low", "Close"]

        for column in columns_to_plot:
            # Создание фигуры
            plt.figure(figsize=(10, 6))

            # Построение гистограммы
            sns.histplot(df[column], kde=True, color="blue", bins=30)
            plt.title(f"Распределение значений ({column})", fontsize=16)
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Частота", fontsize=12)
            plt.grid()

            # Описание статистики
            mean = df[column].mean()
            std = df[column].std()
            min_val = df[column].min()
            max_val = df[column].max()
            quantiles = df[column].quantile([0.25, 0.5, 0.75])
            textstr = (
                f"Mean: {mean:.2f}\n"
                f"Std. Dev.: {std:.2f}\n"
                f"Min: {min_val:.2f}\n"
                f"25%: {quantiles[0.25]:.2f}\n"
                f"Median: {quantiles[0.5]:.2f}\n"
                f"75%: {quantiles[0.75]:.2f}\n"
                f"Max: {max_val:.2f}"
            )

            # Добавление текста с описательной статистикой
            plt.gca().text(
                0.95,
                0.95,
                textstr,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
            )

            # Сохранение графика
            plot_path = os.path.join("output/visualizations", f"{column}_distribution.png")
            plt.savefig(plot_path)
            logging.info(f"График для {column} сохранен: {plot_path}")

            # Отображение графика в консоли
            plt.show()

            # Очистка графика перед следующим построением
            plt.clf()

    except Exception as e:
        logging.error(f"Ошибка при визуализации распределения данных: {e}")
        raise

def plot_predictions(y_true, y_pred, title, filename, metrics=None):
    """
    Построение графика фактических и предсказанных значений с добавлением метрик.

    :param y_true: Фактические значения.
    :param y_pred: Предсказанные значения.
    :param title: Заголовок графика.
    :param filename: Имя файла для сохранения графика.
    :param metrics: Словарь с метриками (например, {"MSE": 0.01, "MAE": 0.02}).
    """
    try:
        # Создание папки для визуализаций
        os.makedirs("output/visualizations", exist_ok=True)

        # Создание графика
        plt.figure(figsize=(12, 6))

        # Построение линий фактических значений и предсказаний
        plt.plot(y_true.values, label="Фактические значения", color="blue")
        plt.plot(y_pred, label="Прогноз", color="orange", linestyle="--")

        # Настройка заголовков и осей
        plt.title(title, fontsize=16)
        plt.xlabel("Индексы", fontsize=14)
        plt.ylabel("Цена закрытия ($)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid()

        # Добавление текста с метриками на график
        if metrics:
            # Форматирование текста метрик
            metrics_text = "\n".join(
                [f"{key}: {value:.4f}" if isinstance(value, (float, int)) else f"{key}: {value}" for key, value in metrics.items()]
            )
            plt.text(
                0.05, 0.95, metrics_text,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="left",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
            )

        # Сохранение графика
        save_path = os.path.join("output/visualizations", filename)
        plt.savefig(save_path)
        plt.show()
        plt.close()
        logging.info(f"График сохранён в {save_path}.")
    except Exception as e:
        logging.error(f"Ошибка при построении графика: {e}")
        raise

def plot_anomalies_with_model(y_true, y_pred, anomalies, title, filename):
    """
    Построение графика аномалий с отображением модели, нормальных наблюдений и выбросов.
    """
    try:
        # Создание индексов аномалий
        anomaly_indices = np.where(anomalies)[0]
        anomaly_indices = anomaly_indices[anomaly_indices < len(y_true)]  # Ограничение индексов

        # Убедитесь, что индексы не выходят за пределы данных
        normal_indices = np.setdiff1d(np.arange(len(y_true)), anomaly_indices)

        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.plot(y_pred, label="Полиномиальная модель", color="green", linewidth=2)
        plt.scatter(normal_indices, y_true.iloc[normal_indices], label="Нормальные наблюдения", color="black", alpha=0.6)
        plt.scatter(anomaly_indices, y_true.iloc[anomaly_indices], label="Выбросы", color="red", zorder=5)

        # Добавление соединительных линий
        for idx in anomaly_indices:
            plt.plot(
                [idx, idx],
                [y_pred[idx], y_true.iloc[idx]],
                color="blue",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

        # Настройка графика
        plt.title(title, fontsize=16)
        plt.xlabel("Индексы", fontsize=14)
        plt.ylabel("Значения", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid()

        # Сохранение графика
        save_path = os.path.join("output/visualizations", filename)
        plt.savefig(save_path)
        plt.show()
        plt.close()
        logging.info(f"График сохранён в {save_path}.")
    except Exception as e:
        logging.error(f"Ошибка при построении графика: {e}")
        raise