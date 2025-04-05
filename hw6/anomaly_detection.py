import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest


def detect_anomalies(y_test: pd.Series, predictions: np.ndarray, threshold: float = 0.1) -> pd.Series:
    """
    Поиск аномалий на основе разницы между реальными значениями и прогнозами.
    
    Args:
        y_test (pd.Series): Реальные значения.
        predictions (np.ndarray): Прогнозируемые значения.
        threshold (float): Порог для определения аномалии.
        
    Returns:
        pd.Series: Аномалии, где разница превышает порог.
    """
    logging.info("Поиск аномалий на основе разницы между фактическими и прогнозируемыми значениями...")
    try:
        # Проверка на пустые данные
        if y_test.empty or len(predictions) == 0:
            raise ValueError("Входные данные для поиска аномалий пустые.")

        if len(y_test) != len(predictions):
            raise ValueError("Длина y_test и predictions не совпадает.")

        # Проверка на наличие NaN
        if y_test.isna().any() or np.isnan(predictions).any():
            raise ValueError("Входные данные содержат NaN значения.")

        # Вычисление остатков (разницы)
        residuals = np.abs(y_test - predictions)

        # Поиск аномалий
        anomalies = y_test[residuals > threshold]

        logging.info(f"Найдено {len(anomalies)} аномалий. Порог: {threshold}")
        return anomalies
    except Exception as e:
        logging.error(f"Ошибка при поиске аномалий: {e}", exc_info=True)
        raise


def detect_anomalies_isolation_forest(
    df: pd.DataFrame, column: str, contamination: float = 0.01
) -> pd.Series:
    """
    Поиск аномалий в данных с использованием Isolation Forest.
    
    Args:
        df (pd.DataFrame): Данные.
        column (str): Название столбца для анализа.
        contamination (float): Доля аномалий в выборке.
        
    Returns:
        pd.Series: Логический массив, где True - это аномалия.
    """
    logging.info(f"Поиск аномалий в столбце '{column}' с использованием Isolation Forest...")
    try:
        # Проверка на наличие столбца
        if column not in df.columns:
            raise KeyError(f"Столбец '{column}' отсутствует в DataFrame.")

        # Проверка на пустой DataFrame
        if df.empty:
            raise ValueError("Передан пустой DataFrame.")

        # Проверка на наличие NaN
        if df[column].isna().any():
            raise ValueError(f"Столбец '{column}' содержит NaN значения.")

        # Обучение Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        data = df[column].values.reshape(-1, 1)
        anomalies = model.fit_predict(data)

        # Возвращаем логический массив, где -1 означает аномалию
        logging.info(f"Isolation Forest обнаружил {np.sum(anomalies == -1)} аномалий.")
        return anomalies == -1
    except KeyError as e:
        logging.error(f"Ошибка: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Ошибка при поиске аномалий с использованием Isolation Forest: {e}", exc_info=True)
        raise


def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Удаление выбросов в указанном столбце с использованием метода межквартильного размаха (IQR).
    
    Args:
        df (pd.DataFrame): Данные.
        column (str): Название столбца, в котором нужно удалить выбросы.
        
    Returns:
        pd.DataFrame: Данные без выбросов.
    """
    logging.info(f"Удаление выбросов для столбца '{column}'...")
    try:
        # Проверка на наличие столбца
        if column not in df.columns:
            raise KeyError(f"Столбец '{column}' отсутствует в DataFrame.")

        # Проверка на пустой DataFrame
        if df.empty:
            raise ValueError("Передан пустой DataFrame.")

        # Вычисление межквартильного размаха (IQR)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Границы для определения выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Фильтруем выбросы
        original_count = len(df)
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        filtered_count = len(filtered_df)

        logging.info(f"Удалено {original_count - filtered_count} выбросов из столбца '{column}'.")
        logging.info(f"Границы выбросов: нижняя = {lower_bound}, верхняя = {upper_bound}.")
        return filtered_df
    except KeyError as e:
        logging.error(f"Ошибка: {e}", exc_info=True)
        raise
    except Exception as e:
        logging.error(f"Ошибка при удалении выбросов: {e}", exc_info=True)
        raise


def process_anomalies(
    df: pd.DataFrame, column: str, anomalies: pd.Series, method: str = "interpolation"
) -> pd.DataFrame:
    """
    Обработка аномалий в данных.
    
    Args:
        df (pd.DataFrame): Данные.
        column (str): Название столбца для обработки.
        anomalies (pd.Series): Логический массив аномалий.
        method (str): Метод обработки ('remove' или 'interpolation').
        
    Returns:
        pd.DataFrame: Данные с обработанными аномалиями.
    """
    logging.info(f"Обработка аномалий в столбце '{column}' методом '{method}'...")
    try:
        if method == "remove":
            logging.info("Удаление аномалий...")
            return df[~anomalies]
        elif method == "interpolation":
            logging.info("Замена аномалий на интерполированные значения...")
            df_cleaned = df.copy()
            df_cleaned.loc[anomalies, column] = np.nan
            df_cleaned[column] = df_cleaned[column].interpolate()
            return df_cleaned
        else:
            raise ValueError("Недопустимый метод обработки аномалий. Используйте 'remove' или 'interpolation'.")
    except Exception as e:
        logging.error(f"Ошибка при обработке аномалий: {e}", exc_info=True)
        raise