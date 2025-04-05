from sklearn.metrics import mean_squared_error
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Вычисление метрик для модели.
    :param y_true: Фактические значения.
    :param y_pred: Предсказанные значения.
    :return: Словарь метрик.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE в процентах

    return {
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "MAPE": mape
    }

def train_models(X_train, y_train, X_test, y_test):
    """
    Обучение моделей машинного обучения (XGBoost, LightGBM, CatBoost).
    """
    logging.info("Обучение моделей машинного обучения...")
    models = {
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
    }

    results = {}
    metrics = {}
    for name, model in models.items():
        logging.info(f"Обучение модели: {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Расчёт метрик
        model_metrics = calculate_metrics(y_test, predictions)
        metrics[name] = model_metrics
        
        logging.info(f"{name} метрики: {model_metrics}")
        results[name] = predictions

    # Усреднение предсказаний моделей машинного обучения
    ensemble_ml = np.mean(list(results.values()), axis=0)
    ensemble_metrics = calculate_metrics(y_test, ensemble_ml)
    metrics["Ensemble ML"] = ensemble_metrics
    logging.info(f"Ансамбль ML моделей метрики: {ensemble_metrics}")

    # Сохранение метрик
    save_metrics(metrics)

    return ensemble_ml, metrics


def train_arima(y_train, y_test):
    """
    Обучение ARIMA модели.
    """
    logging.info("Обучение ARIMA модели...")
    model = ARIMA(y_train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(y_test))

    # Расчёт метрик
    arima_metrics = calculate_metrics(y_test, predictions)
    logging.info(f"ARIMA метрики: {arima_metrics}")

    # Сохранение метрик
    save_metrics({"ARIMA": arima_metrics})

    return predictions, {"ARIMA": arima_metrics}


def train_lstm(X_train, y_train, X_test, y_test):
    """
    Обучение LSTM модели.
    """
    logging.info("Обучение LSTM модели...")

    # Преобразование данных для LSTM
    X_train_lstm = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(X_train_lstm.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)

    predictions = model.predict(X_test_lstm).flatten()

    # Расчёт метрик
    lstm_metrics = calculate_metrics(y_test, predictions)
    logging.info(f"LSTM метрики: {lstm_metrics}")

    # Сохранение метрик
    save_metrics({"LSTM": lstm_metrics})

    return predictions, {"LSTM": lstm_metrics}


def create_ensemble(predictions_list, y_test):
    """
    Создание ансамбля на основе предсказаний всех моделей.
    """
    logging.info("Создание ансамбля моделей...")
    ensemble_predictions = np.mean(predictions_list, axis=0)

    # Расчёт метрик
    ensemble_metrics = calculate_metrics(y_test, ensemble_predictions)
    logging.info(f"Ансамбль всех моделей метрики: {ensemble_metrics}")

    # Сохранение метрик
    save_metrics({"Ensemble All": ensemble_metrics})

    return ensemble_predictions, {"Ensemble All": ensemble_metrics}

def save_metrics(metrics, file_name="output/metrics.txt"):
    """
    Сохраняет метрики в файл.
    """
    os.makedirs("output", exist_ok=True)
    with open(file_name, "a") as f:
        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name}:\n")
            for metric_name, value in model_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")
