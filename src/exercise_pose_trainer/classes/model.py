import abc

from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV


class Model(abc.ABC):
    def __init__(self) -> None:
        # super().__init__()
        self._model: BaseEstimator
        self._param_grid: dict[str, list[str | int | float]]

        if type(self) is Model:
            raise TypeError(
                'Model is an abstract class and cannot be instantiated directly.')

    def fit(self, X, y) -> None:
        grid_search = GridSearchCV(self._model, self._param_grid, n_jobs=-1)
        grid_search.fit(X, y)
        self._model = grid_search.best_estimator_

    def get_params(self) -> dict:
        return self._model.get_params()

    def predict(self, X):
        return self._model.predict(X)  # type: ignore


class ModelFactory:
    @staticmethod
    def get_model(model_type: str) -> Model:
        if model_type == 'svm':
            return SVMModel()
        else:
            raise ValueError(f'Unknown model type: {model_type}')


class SVMModel(Model):
    def __init__(self):
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
