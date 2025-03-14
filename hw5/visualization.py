import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import skew, kurtosis


def ensure_output_folder(output_folder):
    """Проверяет существование папки и создает ее, если необходимо."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def plot_target_distribution(y, output_folder, show=False):
    """
    Визуализация распределения целевой переменной с добавлением текстовых метрик (среднее, медиана и стандартное отклонение).
    """
    # Проверка папки
    ensure_output_folder(output_folder)

    # Вычисление метрик
    mean_value = np.mean(y)
    median_value = np.median(y)
    std_value = np.std(y)

    # Построение графика
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=30, color="blue", alpha=0.7)
    plt.title("Распределение целевой переменной", fontsize=16)
    plt.xlabel("Значение целевой переменной", fontsize=12)
    plt.ylabel("Частота", fontsize=12)

    # Добавление текстовых метрик на график
    text_str = f"Среднее: {mean_value:.2f}\nМедиана: {median_value:.2f}\nСт. отклонение: {std_value:.2f}"
    plt.gca().text(
        0.95, 0.95, text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black")
    )

    # Сохранение графика
    output_path = os.path.join(output_folder, "target_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, output_folder, model_name, mse, mae, r2, show=False):
    """
    Визуализация предсказаний модели против истинных значений с текстовыми характеристиками.
    """
    # Проверка папки
    ensure_output_folder(output_folder)

    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label="Предсказания")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2, label="Идеальная линия")
    plt.title(f"Предсказания vs Истинные значения ({model_name})")
    plt.xlabel("Истинные значения")
    plt.ylabel("Предсказания")
    plt.legend()
    plt.grid(True)

    # Добавление текстовых характеристик
    text = (
        f"R²: {r2:.4f}\n"
        f"MSE: {mse:.4f}\n"
        f"MAE: {mae:.4f}"
    )
    plt.text(
        0.05, 0.95, text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    output_path = os.path.join(output_folder, f"predictions_vs_actual_{model_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(y_true, y_pred, output_folder, model_name, show=False):
    """
    Визуализация распределения остатков.
    """
    # Проверка папки
    ensure_output_folder(output_folder)

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='green')
    plt.title(f"Распределение остатков ({model_name})")
    plt.xlabel("Остатки")
    plt.ylabel("Частота")
    plt.grid(True)

    output_path = os.path.join(output_folder, f"residuals_distribution_{model_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals_vs_predictions(y_true, y_pred, output_folder, model_name, show=False):
    """
    Визуализация остатков против предсказанных значений с текстовыми характеристиками.
    """
    # Проверка папки
    ensure_output_folder(output_folder)

    residuals = y_true - y_pred
    mse_residuals = np.mean(residuals**2)
    mae_residuals = np.mean(np.abs(residuals))
    skewness = skew(residuals)
    kurt = kurtosis(residuals)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5, color='blue')
    plt.axhline(y=0, color='orange', linestyle='--', linewidth=2, label="Остатки == 0")
    plt.title(f"Остатки vs Предсказания ({model_name})")
    plt.xlabel("Предсказания")
    plt.ylabel("Остатки")
    plt.grid(True)

    # Добавление текстовых характеристик
    text = (
        f"MSE остатков: {mse_residuals:.4f}\n"
        f"MAE остатков: {mae_residuals:.4f}\n"
        f"Асимметрия: {skewness:.4f}\n"
        f"Куртозис: {kurt:.4f}"
    )
    plt.text(
        0.05, 0.95, text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    output_path = os.path.join(output_folder, f"residuals_vs_predictions_{model_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(model, feature_names, output_folder, model_name, show=False):
    """
    Визуализация важности признаков для моделей, поддерживающих feature_importances_.
    """
    # Проверка папки
    ensure_output_folder(output_folder)

    try:
        # Извлечение важности признаков
        importances = model.feature_importances_
    except AttributeError:
        raise ValueError(f"Модель {model_name} не поддерживает атрибут feature_importances_.")

    # Проверка длины важности признаков и названий
    if len(importances) != len(feature_names):
        raise ValueError("Длина важностей признаков не совпадает с количеством признаков.")

    # Построение графика
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(importances)[::-1]  # Сортировка по убыванию
    plt.barh(range(len(importances)), importances[sorted_indices], align="center")
    plt.yticks(range(len(importances)), np.array(feature_names)[sorted_indices])
    plt.xlabel("Важность признаков")
    plt.title(f"Важность признаков ({model_name})")
    plt.gca().invert_yaxis()

    # Сохранение графика
    output_path = os.path.join(output_folder, f"{model_name}_feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()