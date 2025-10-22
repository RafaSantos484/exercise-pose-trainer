import argparse
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .classes.utils import Utils
from .classes.landmarker import Landmarker
from .classes.model import ModelFactory


def load_features(base_path: str, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = os.path.join(base_path, 'features.joblib')
    if os.path.exists(cache_path):
        print('Loading cached features...')
        return joblib.load(cache_path)

    y = []
    X_imgs_paths = []
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if not os.path.isdir(label_path):
            continue

        imgs_paths = []
        for f in os.listdir(label_path):
            f_path = os.path.join(label_path, f)
            if Utils.is_img_file(f_path):
                imgs_paths.append(f_path)
        X_imgs_paths.extend(imgs_paths)
        y.extend([label] * len(imgs_paths))

    X_train_imgs_paths, X_test_imgs_paths, y_train, y_test = train_test_split(
        X_imgs_paths, y, test_size=0.3, random_state=seed)

    print('Extracting features from training images...')
    X_train, y_train, _ = Landmarker.get_angles_features_from_imgs(
        X_train_imgs_paths, y=y_train, augment_data=True)
    print('Extracting features from testing images...')
    X_test, y_test, _ = Landmarker.get_angles_features_from_imgs(
        X_test_imgs_paths, y=y_test, augment_data=False)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    data = (X_train, X_test, y_train, y_test)
    joblib.dump(data, cache_path)
    return data


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

    X_train, X_test, y_train, y_test = load_features(args.path, seed=args.seed)
    validation_data = (X_test, y_test)

    model = ModelFactory.get_model(args.model)
    model.fit(X_train, y_train, validation_data=validation_data)
    model.generate_report(X_test, y_test)
    model.view_report()
    model.save_model()
