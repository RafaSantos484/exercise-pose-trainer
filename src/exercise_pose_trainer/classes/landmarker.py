import itertools
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

from exercise_pose_trainer.classes.point3d import Point3d
from exercise_pose_trainer.classes.utils import Utils

_LANDMARKS_DICT: dict[str, int] = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32,
}

_POINTS_TRIPLETS = [('left_wrist', 'left_elbow', 'left_shoulder'),
                    ('right_wrist', 'right_elbow', 'right_shoulder'),
                    ('left_wrist', 'left_shoulder', 'right_shoulder'),
                    ('right_wrist', 'right_shoulder', 'left_shoulder'),

                    ('left_wrist', 'left_shoulder', 'left_hip'),
                    ('right_wrist', 'right_shoulder', 'right_hip'),

                    ('left_shoulder', 'left_hip', 'left_knee'),
                    ('right_shoulder', 'right_hip', 'right_knee'),
                    ('left_hip', 'left_knee', 'left_ankle'),
                    ('right_hip', 'right_knee', 'right_ankle'),
                    ('left_ankle', 'left_hip', 'right_hip'),
                    ('right_ankle', 'right_hip', 'left_hip'),

                    ('left_foot_index', 'left_wrist', 'left_shoulder'),
                    ('right_foot_index', 'right_wrist', 'right_shoulder'),
                    ('left_foot_index', 'left_wrist', 'right_wrist'),
                    ('right_foot_index', 'right_wrist', 'left_wrist'),
                    ]


class Landmarker:
    _model = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=open(
                'pose_landmarker_full.task', 'rb').read()),
            # base_options=python.BaseOptions(model_asset_buffer=open('pose_landmarker_heavy.task', 'rb').read()),
            running_mode=vision.RunningMode.IMAGE,
        )
    )

    @classmethod
    def get_points_from_img_path(cls, image_path: str,
                                 mirror=False,
                                 rotation: float | None = None,
                                 scale: float | None = None,
                                 ) -> list[Point3d] | None:
        image = cv2.imread(image_path)
        if mirror:
            image = cv2.flip(image, 1)
        if rotation is not None:
            image = Utils.rotate_img(image, rotation)
        if scale is not None:
            image = Utils.scale_img(image, scale)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = cls._model.detect(mp_image)
        if not result.pose_world_landmarks:
            print(
                f'No landmarks found for {image_path} (mirror={mirror}, rotation={rotation}, scale={scale})')
            return None

        landmarks = result.pose_world_landmarks[0]
        points = [Point3d.from_landmark(lm) for lm in landmarks]
        return points

    @classmethod
    def get_angles_features_from_imgs(cls, imgs_paths: list[str], y: list[str] | None = None, augment_data=False) -> tuple[list[list[float]], list[str], list[str]]:
        X = []
        y_sucessful = []
        sucessful_img_paths = []
        for i, img_path in enumerate(imgs_paths):
            if not Utils.is_img_file(img_path):
                continue

            mirror_options = [False, True] if augment_data else [False]
            rotation_options = [-15, -5, None, 5,
                                15] if augment_data else [None]
            scale_options = [0.9, None, 1.1] if augment_data else [None]
            data_augmentations_options = itertools.product(
                mirror_options, rotation_options, scale_options)
            for mirror, rotation, scale in data_augmentations_options:
                points = cls.get_points_from_img_path(
                    img_path,
                    mirror=mirror,
                    rotation=rotation,
                    scale=scale,
                )

                if points is None:
                    continue

                angles = []
                for p1_name, p2_name, p3_name in _POINTS_TRIPLETS:
                    p1 = points[_LANDMARKS_DICT[p1_name]]
                    p2 = points[_LANDMARKS_DICT[p2_name]]
                    p3 = points[_LANDMARKS_DICT[p3_name]]
                    angle = Point3d.get_angle_between(p1, p2, p3)
                    angles.append(angle)

                X.append(angles)
                if not augment_data:
                    sucessful_img_paths.append(img_path)
                if y is not None:
                    y_sucessful.append(y[i])

        return X, y_sucessful, sucessful_img_paths
