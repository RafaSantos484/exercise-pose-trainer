import argparse
import os

from .classes.landmarker import Landmarker
from .classes.model import ModelFactory
from .classes.utils import Utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="Path to the test images folder")
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
    args = parser.parse_args()

    imgs_paths = [os.path.join(args.path, f) for f in os.listdir(args.path)]
    X, _, imgs_paths = Landmarker.get_angles_features_from_imgs(imgs_paths)

    model = ModelFactory.load_model(args.model)
    y_pred = model.predict(X)
    for img_path, pred in zip(imgs_paths, y_pred):
        img_file = Utils.get_basename(img_path)
        print(f'{img_file}: {pred}')
