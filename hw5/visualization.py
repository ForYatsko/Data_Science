import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import skew, kurtosis


def plot_target_distribution(y, output_folder, show=False):
    """
    Визуализация распределения целевой переменной.
    """
    plt.figure(figsize=(10, 8))
    sns.histplot(y, kde=True, bins=30, color='blue')
    plt.title("Распределение целевой переменной")
    plt.xlabel("Значение целевой переменной")
    plt.ylabel("Частота")
    plt.grid(True)

    output_path = os.path.join(output_folder, "target_distribution.png")
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions_vs_actual(y_true, y_pred, output_folder, model_name, mse, mae, r2, show=False):
    """
    Визуализация предсказаний модели против истинных значений с текстовыми характеристиками.
    """
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
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals(y_true, y_pred, output_folder, model_name, show=False):
    """
    Визуализация распределения остатков.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='green')
    plt.title(f"Распределение остатков ({model_name})")
    plt.xlabel("Остатки")
    plt.ylabel("Частота")
    plt.grid(True)

    output_path = os.path.join(output_folder, f"residuals_distribution_{model_name}.png")
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_residuals_vs_predictions(y_true, y_pred, output_folder, model_name, show=False):
    """
    Визуализация остатков против предсказанных значений с текстовыми характеристиками.
    """
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
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(importances, feature_names, output_folder, model_name, show=False):
    """
    Визуализация важности признаков модели.
    """
    feature_importance = np.array(importances)
    feature_names = np.array(feature_names)

    indices = np.argsort(feature_importance)[::-1]
    sorted_importance = feature_importance[indices]
    sorted_features = feature_names[indices]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis")
    plt.title(f"Важность признаков ({model_name})")
    plt.xlabel("Важность")
    plt.ylabel("Признаки")
    plt.grid(True)

    output_path = os.path.join(output_folder, f"feature_importance_{model_name}.png")
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()


def plot_residual_statistics(y_true, y_pred, output_folder, model_name, show=False):
    """
    Визуализация числовых характеристик распределения остатков.
    """
    # Вычисление метрик распределения остатков
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    skewness = skew(residuals)
    kurt = kurtosis(residuals)
    outliers_ratio = np.mean(np.abs(residuals) > 2 * std_residual)

    # Построение текста для графика
    text = (
        f"Среднее остатков: {mean_residual:.4f}\n"
        f"Стандартное отклонение: {std_residual:.4f}\n"
        f"Асимметрия: {skewness:.4f}\n"
        f"Куртозис: {kurt:.4f}\n"
        f"Доля выбросов (>2σ): {outliers_ratio:.2%}"
    )

    # Построение гистограммы остатков с текстовыми выводами
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='purple')
    plt.title(f"Распределение остатков ({model_name}) с метриками")
    plt.xlabel("Остатки")
    plt.ylabel("Частота")
    plt.grid(True)

    # Добавление текста на график
    plt.text(
        0.95, 0.95, text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Сохранение графика
    output_path = os.path.join(output_folder, f"residual_statistics_{model_name}.png")
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()