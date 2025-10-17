import argparse
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .classes.landmarker import Landmarker
from .classes.model import ModelFactory


def load_features(base_path: str, augment_data=False) -> tuple[list[list[float]], list[str]]:
    cache_path = os.path.join(base_path, 'features.joblib')
    if os.path.exists(cache_path):
        print('Loading cached features...')
        return joblib.load(cache_path)

    X = []
    y = []
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if not os.path.isdir(label_path):
            continue

        imgs_paths = [os.path.join(label_path, f)
                      for f in os.listdir(label_path)]
        label_features, _ = Landmarker.get_angles_features_from_imgs(
            imgs_paths, augment_data=augment_data)
        X.extend(label_features)
        y.extend([label] * len(label_features))

    joblib.dump((X, y), cache_path)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the features folder')
    parser.add_argument(
        'model',
        type=str,
        choices=['fcnn',
                 'gradient_boosting',
                 'logistic_regression',
                 'random_forest',
                 'svm'],
        help='Model type to use',
    )
    parser.add_argument('--seed', type=int,
                        help='Seed used for reproducibility')

    args = parser.parse_args()

    X, y = load_features(args.path, augment_data=True)
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed)

    model = ModelFactory.get_model(args.model)
    print(model._name)
    model.fit(X_train, y_train)
    params = model.get_params()
    print(params)
    model.generate_report(X_test, y_test)
    model.save_model()
