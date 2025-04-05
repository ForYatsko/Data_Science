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


def save_metrics(metrics, file_name="output/metrics.txt"):
    """
    Сохраняет метрики в файл.
    """
    try:
        with open(file_name, "a") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        logging.info(f"Метрики успешно сохранены в файл {file_name}.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении метрик: {e}")


def train_models(X_train, y_train, X_test, y_test):
    """
    Обучение моделей машинного обучения (XGBoost, LightGBM, CatBoost).
    Возвращает предсказания ансамбля и метрики.
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
        try:
            logging.info(f"Обучение модели: {name}...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            logging.info(f"{name} MSE: {mse}")
            results[name] = predictions
            metrics[name] = {"MSE": mse}
        except Exception as e:
            logging.error(f"Ошибка при обучении модели {name}: {e}")
            results[name] = None
            metrics[name] = {"MSE": None}

    # Усреднение предсказаний моделей машинного обучения
    valid_predictions = [pred for pred in results.values() if pred is not None]
    if valid_predictions:
        ensemble_ml = np.mean(valid_predictions, axis=0)
        mse_ensemble = mean_squared_error(y_test, ensemble_ml)
        logging.info(f"Ансамбль ML моделей MSE: {mse_ensemble}")
        metrics["Ensemble ML"] = {"MSE": mse_ensemble}
    else:
        logging.error("Не удалось создать ансамбль. Нет валидных предсказаний.")
        ensemble_ml = None
        metrics["Ensemble ML"] = {"MSE": None}

    # Сохранение метрик
    save_metrics(metrics)

    return ensemble_ml, metrics


def train_arima(y_train, y_test):
    """
    Обучение ARIMA модели.
    Возвращает предсказания и метрики.
    """
    try:
        logging.info("Обучение ARIMA модели...")
        model = ARIMA(y_train, order=(5, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(y_test))
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"ARIMA MSE: {mse}")
        save_metrics({"ARIMA MSE": mse})
        return predictions, {"MSE": mse}
    except Exception as e:
        logging.error(f"Ошибка при обучении ARIMA модели: {e}")
        return None, {"MSE": None}


def train_lstm(X_train, y_train, X_test, y_test):
    """
    Обучение LSTM модели.
    Возвращает предсказания и метрики.
    """
    try:
        logging.info("Обучение LSTM модели...")

        # Преобразование данных для LSTM
        X_train_lstm = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_lstm = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential([
            LSTM(50, activation="relu", input_shape=(X_train_lstm.shape[1], 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=1)

        predictions = model.predict(X_test_lstm).flatten()
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"LSTM MSE: {mse}")
        save_metrics({"LSTM MSE": mse})
        return predictions, {"MSE": mse}
    except Exception as e:
        logging.error(f"Ошибка при обучении LSTM модели: {e}")
        return None, {"MSE": None}

def create_ensemble(predictions_list, y_test=None):
    """
    Создание ансамбля на основе предсказаний всех моделей.
    Возвращает предсказания ансамбля и метрики (если передан y_test).
    """
    try:
        logging.info("Создание ансамбля моделей...")
        valid_predictions = [pred for pred in predictions_list if pred is not None]
        if valid_predictions:
            ensemble_predictions = np.mean(valid_predictions, axis=0)
            metrics = {}
            if y_test is not None:
                mse_ensemble = mean_squared_error(y_test, ensemble_predictions)
                metrics["MSE"] = mse_ensemble
                logging.info(f"Ансамбль MSE: {mse_ensemble}")
            return ensemble_predictions, metrics
        else:
            logging.error("Невозможно создать ансамбль: нет валидных предсказаний.")
            return None, {"MSE": None}
    except Exception as e:
        logging.error(f"Ошибка при создании ансамбля: {e}")
        return None, {"MSE": None}
