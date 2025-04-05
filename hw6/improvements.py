import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from anomaly_detection import detect_anomalies_isolation_forest, process_anomalies


# Улучшение данных
def preprocess_with_anomalies(df, column, method="interpolation", contamination=0.01):
    """
    Обработка данных с учётом аномалий, используя Isolation Forest.
    
    Args:
        df (pd.DataFrame): Данные.
        column (str): Название столбца для обработки.
        method (str): Метод обработки ('remove' или 'interpolation').
        contamination (float): Доля аномалий в данных.
        
    Returns:
        pd.DataFrame: Данные с обработанными аномалиями.
    """
    # Выявление аномалий
    anomalies = detect_anomalies_isolation_forest(df, column, contamination)
    
    # Обработка аномалий
    cleaned_df = process_anomalies(df, column, anomalies, method=method)
    return cleaned_df


# Улучшение ансамбля
def weighted_ensemble(predictions, weights):
    """
    Создаёт взвешенный ансамбль на основе предсказаний моделей.
    
    Args:
        predictions (list of np.array): Список предсказаний моделей.
        weights (list of float): Веса для каждой модели.
        
    Returns:
        np.array: Итоговые предсказания ансамбля.
    """
    predictions = np.array(predictions)
    weights = np.array(weights)
    return np.average(predictions, axis=0, weights=weights)