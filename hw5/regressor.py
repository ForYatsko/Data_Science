from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    BayesianRidge,
    HuberRegressor,
    ElasticNet,
    PassiveAggressiveRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logger = logging.getLogger(__name__)

def build_regressors():
    """Создание и настройка всех регрессоров с оптимальными параметрами."""
    logger.info("Инициализация списка регрессоров.")

    return {
        # Ensemble models
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "LGBM": LGBMRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=-1, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6, verbosity=0, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=42
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=42
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),

        # Linear models
        "Lasso": Lasso(
            alpha=0.1, max_iter=1000, random_state=42
        ),
        "Ridge": Ridge(
            alpha=1.0, random_state=42
        ),
        "Bayesian Ridge": BayesianRidge(),
        "Linear Regression": LinearRegression(),
        "Huber": HuberRegressor(
            epsilon=1.35, max_iter=100, alpha=0.0001
        ),
        "Elastic Net": ElasticNet(
            alpha=1.0, l1_ratio=0.5, max_iter=1000, random_state=42
        ),
        "Passive Aggressive": PassiveAggressiveRegressor(
            max_iter=1000, tol=1e-4, random_state=42
        ),

        # Tree-based models
        "Decision Tree": DecisionTreeRegressor(
            max_depth=6, random_state=42
        ),

        # Neighbors-based models
        "KNN": KNeighborsRegressor(
            n_neighbors=5, weights="distance", algorithm="auto"
        ),

        # Baseline model
        "Dummy": DummyRegressor(strategy="mean"),
    }


def evaluate_regressors(regressors, X_train, X_test, y_train, y_test):
    """Обучение и оценка всех регрессоров."""
    logger.info("Начало обучения и оценки регрессоров.")
    results = []
    for name, model in regressors.items():
        logger.info(f"Обучение регрессора: {name}")
        
        try:
            # Обучение модели
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            
            # Метрики
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Логирование результатов
            logger.info(f"Результаты регрессора {name} — MSE: {mse}, MAE: {mae}, R2: {r2}")
            
            # Добавление результатов в список
            results.append({
                "Regressor": name,
                "MSE": mse,
                "MAE": mae,
                "R2": r2,
            })
        
        except Exception as e:
            # Логирование ошибок
            logger.error(f"Ошибка при работе с регрессором {name}: {e}")
            results.append({
                "Regressor": name,
                "MSE": None,
                "MAE": None,
                "R2": None,
                "Error": str(e),
            })

    logger.info("Оценка регрессоров завершена.")
    return results