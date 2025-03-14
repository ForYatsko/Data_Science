from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def prepare_data(df, target_column):
    """Подготовка данных: разделение на признаки и целевые переменные, стандартизация."""
    logger.info(f"Подготовка данных. Целевая переменная: {target_column}.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    
    # Стандартизация и преобразование обратно в DataFrame
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Добавляем обратно названия колонок
    
    logger.info("Данные успешно стандартизированы.")
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Разделение данных на обучающую и тестовую выборки."""
    logger.info(f"Разделение данных на обучающие и тестовые выборки. Test size: {test_size}.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info("Данные успешно разделены.")
    return X_train, X_test, y_train, y_test