import argparse
import os

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .classes.model import ModelFactory
from .classes.point3d import Point3d
from .classes.utils import Utils
from .classes.landmarker import Landmarker, landmarks_dict


def load_features(base_path: str):
    labels = []
    features = []
    points_triplets = [('left_wrist', 'left_elbow', 'left_shoulder'),
                       ('right_wrist', 'right_elbow', 'right_shoulder'),
                       ('left_elbow', 'left_shoulder', 'right_shoulder'),
                       ('right_elbow', 'right_shoulder', 'left_shoulder'),
                       ]

    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if not os.path.isdir(label_path):
            continue

        # for feature_file in os.listdir(label_path)[:10]:
        for feature_file in os.listdir(label_path):
            if not Utils.is_img_file(feature_file):
                continue

            img_points = Landmarker.get_points(
                os.path.join(label_path, feature_file), mirror=False)
            if img_points is not None:
                angles = []
                for p1_name, p2_name, p3_name in points_triplets:
                    p1 = img_points[landmarks_dict[p1_name]]
                    p2 = img_points[landmarks_dict[p2_name]]
                    p3 = img_points[landmarks_dict[p3_name]]
                    angle = Point3d.get_angle_between(p1, p2, p3)
                    angles.append(angle)

                features.append(angles)
                labels.append(label)

    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the features folder')
    parser.add_argument(
        'model',
        type=str,
        choices=['random_forest', 'svm'],
        help='Model type to use. Choices are: "svm" or "random_forest"',
    )
    parser.add_argument('--seed', type=int,
                        help='Seed used for reproducibility')

    args = parser.parse_args()

    features, labels = load_features(args.path)
    # print(features[:5])

    X, y = np.array(features), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed)
    model = ModelFactory.get_model(args.model)
    model.fit(X_train, y_train)
    params = model.get_params()
    print(params)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
