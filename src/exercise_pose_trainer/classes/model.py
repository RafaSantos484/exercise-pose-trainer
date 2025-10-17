import abc
import os

import joblib
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class Model(abc.ABC):
    def __init__(self) -> None:
        # super().__init__()
        self._name: str
        self._report: str
        self._cm: np.ndarray
        self._model: BaseEstimator
        self._param_grid: dict[str, list[None | str | int | float]]
        self._shape: tuple[int, ...]

        if type(self) is Model:
            raise TypeError(
                'Model is an abstract class and cannot be instantiated directly.')

    def fit(self, X, y) -> None:
        X, y = np.array(X), np.array(y)
        self._shape = X.shape[1:]

        grid_search = GridSearchCV(self._model, self._param_grid, n_jobs=-1)
        grid_search.fit(X, y)
        self._model = grid_search.best_estimator_

    def get_params(self) -> dict:
        return self._model.get_params()

    def predict(self, X) -> list:
        return self._model.predict(X)  # type: ignore

    def generate_report(self, X_test, y_test) -> None:
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        print(report)
        print(cm)

        self._report = str(report)
        self._cm = cm

    def save_model(self) -> None:
        os.makedirs(os.path.join('models', self._name), exist_ok=True)
        model_path = os.path.join('models', self._name)

        joblib.dump(self, os.path.join(model_path, 'model.joblib'))

        shape = [None, *self._shape]
        initial_type = [('input', FloatTensorType(shape))]
        onnx_model = convert_sklearn(self._model, initial_types=initial_type)
        if isinstance(onnx_model, tuple):
            onnx_model = onnx_model[0]
        with open(os.path.join(model_path, 'model.onnx'), 'wb') as f:
            f.write(onnx_model.SerializeToString())


class ModelFactory:
    @staticmethod
    def get_model(model_type: str) -> Model:
        if model_type == 'gradient_boosting':
            return _GradientBoostingModel()
        elif model_type == 'logistic_regression':
            return _LogisticRegressionModel()
        elif model_type == 'random_forest':
            return _RandomForestModel()
        elif model_type == 'svm':
            return _SVMModel()
        else:
            raise ValueError(f'Unknown model type: {model_type}')

    @staticmethod
    def load_model(model_type: str) -> Model:
        model_path = os.path.join('models', model_type, 'model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        model = joblib.load(model_path)
        return model


class _GradientBoostingModel(Model):
    def __init__(self):
        self._name = 'gradient_boosting'
        self._model = GradientBoostingClassifier()
        self._param_grid = {
            "learning_rate": [0.01, 0.1, 0.5, 1],
            "n_estimators": [50, 100, 200, 300, 500],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_split": [2, 5, 10, 20, 50],
            "max_depth": [3, 5, 10]
        }


class _LogisticRegressionModel(Model):
    def __init__(self):
        self._name = 'logistic_regression'
        self._model = LogisticRegression()
        self._param_grid = {
            "penalty": [None, "l1", "l2", "elasticnet"],
            "C": [0.01, 0.1, 1, 10, 50, 100],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
            "max_iter": [None, 10, 50, 100, 200]
        }


class _RandomForestModel(Model):
    def __init__(self):
        self._name = 'random_forest'
        self._model = RandomForestClassifier()
        self._param_grid = {
            "n_estimators": [10, 50, 100, 200],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 10, 20, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5, 10]
        }


class _SVMModel(Model):
    def __init__(self):
        self._name = 'svm'
        self._model = svm.SVC()
        '''
        self._param_grid = {
            'C': [0.01, 0.1, 1, 10, 50, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5, 10],
            'gamma': ['scale', 'auto'],
            'coef0': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
            'shrinking': [True, False],
        }
        '''
        self._param_grid = {
            'C': [0.01, 0.1, 1, 10, 50],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5, 10],
            'gamma': ['scale', 'auto']
        }
