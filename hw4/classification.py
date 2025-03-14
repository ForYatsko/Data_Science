from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class ClassifierComparison:
    def __init__(self, X_train, X_test, y_train, y_test, use_pca=False, n_components=10):
        """
        Инициализация сравнения классификаторов.

        :param X_train: Данные для обучения.
        :param X_test: Данные для тестирования.
        :param y_train: Метки классов для обучения.
        :param y_test: Метки классов для тестирования.
        :param use_pca: Применять ли PCA для уменьшения размерности.
        :param n_components: Количество компонент при использовании PCA.
        """
        if X_train.shape[1] == 0:
            raise ValueError("X_train не содержит признаков. Проверьте входные данные.")

        # Применение PCA, если указано
        if use_pca:
            n_components = min(n_components, X_train.shape[1])
            if n_components == 0:
                raise ValueError("Недостаточно признаков для применения PCA.")
            pca = PCA(n_components=n_components)
            self.X_train = pca.fit_transform(X_train)
            self.X_test = pca.transform(X_test)
        else:
            self.X_train = X_train
            self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        # Для хранения результатов
        self.results = []

        # Список классификаторов
        self.classifiers = {
            "GradientBoosting": GradientBoostingClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "AdaBoost": AdaBoostClassifier(),
            "ExtraTrees": ExtraTreesClassifier(),
            "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
            "LightGBM": LGBMClassifier(min_gain_to_split=0.2, min_data_in_leaf=50, verbose=-1),
            "KNeighbors": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss', verbosity=0),
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "SVM (Linear Kernel)": SVC(kernel="linear", probability=True),
        }

    def evaluate_classifiers(self):
        """
        Обучает классификаторы и оценивает их на тестовых данных.
        """
        self.results = []  # Сброс результатов перед началом
        for name, clf in self.classifiers.items():
            print(f"Обучение классификатора: {name}")
            try:
                # Обучение модели
                clf.fit(self.X_train, self.y_train)

                # Предсказания на тестовых данных
                y_pred = clf.predict(self.X_test)
                y_proba = clf.predict_proba(self.X_test)[:, 1] if hasattr(clf, "predict_proba") else None

                # Вычисление метрик
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average="weighted")
                recall = recall_score(self.y_test, y_pred, average="weighted")
                roc_auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else None

                self.results.append({
                    "Classifier": name,
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Recall": recall,
                    "ROC-AUC": roc_auc if roc_auc is not None else "N/A"
                })
            except Exception as e:
                print(f"Ошибка при обучении классификатора {name}: {e}")

        # Проверяем, добавлены ли результаты
        if not self.results:
            raise RuntimeError("Ни один классификатор не был успешно обучен. Проверьте данные и модели.")

    def get_results(self):
        """
        Возвращает результаты в виде DataFrame.
        """
        if not self.results:
            raise ValueError("Результаты пусты. Сначала выполните evaluate_classifiers().")
        results_df = pd.DataFrame(self.results)

        # Проверка структуры DataFrame
        required_columns = ["Classifier", "Accuracy", "F1 Score", "Recall", "ROC-AUC"]
        missing_columns = [col for col in required_columns if col not in results_df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы: {missing_columns}")
        return results_df