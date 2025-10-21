import argparse

from .classes.model import ModelFactory


def main():
    parser = argparse.ArgumentParser()
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

    model = ModelFactory.load_model(args.model)
    model.view_report()
