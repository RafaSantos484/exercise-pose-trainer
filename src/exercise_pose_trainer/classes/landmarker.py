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
                    ('left_elbow', 'left_shoulder', 'right_shoulder'),
                    ('right_elbow', 'right_shoulder', 'left_shoulder'),

                    ('left_shoulder', 'left_hip', 'left_knee'),
                    ('right_shoulder', 'right_hip', 'right_knee'),
                    ('left_shoulder', 'left_hip', 'right_hip'),
                    ('right_shoulder', 'right_hip', 'left_hip'),

                    ('left_hip', 'left_knee', 'left_ankle'),
                    ('right_hip', 'right_knee', 'right_ankle'),
                    ('left_knee', 'left_ankle', 'right_ankle'),
                    ('right_knee', 'right_ankle', 'left_ankle'),

                    ('left_wrist', 'left_elbow', 'left_hip'),
                    ('right_wrist', 'right_elbow', 'right_hip'),
                    ('left_wrist', 'left_elbow', 'left_knee'),
                    ('right_wrist', 'right_elbow', 'right_knee'),
                    ('left_wrist', 'left_elbow', 'left_ankle'),
                    ('right_wrist', 'right_elbow', 'right_ankle'),
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
                            rotation: float | None = None
                            ) -> list[Point3d] | None:
        image = cv2.imread(image_path)
        if mirror:
            image = cv2.flip(image, 1)
        if rotation is not None:
            image = Utils.rotate_img(image, rotation)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = cls._model.detect(mp_image)
        if not result.pose_world_landmarks:
            print(f'No landmarks found for {image_path}')
            return None

        landmarks = result.pose_world_landmarks[0]
        points = [Point3d.from_landmark(lm) for lm in landmarks]
        return points

    @classmethod
    def get_angles_features_from_imgs(cls, imgs_paths: list[str], augment_data=False) -> tuple[list[list[float]], list[str]]:
        X = []
        sucessful_img_paths = []
        for img_path in imgs_paths:
            if not Utils.is_img_file(img_path):
                continue

            mirror_options = [False, True] if augment_data else [False]
            rotation_options = [-10, -5, 0, 5, 10] if augment_data else [0]
            for mirror in mirror_options:
                for rotation in rotation_options:
                    points = cls.get_points_from_img_path(
                        img_path,
                        mirror=mirror,
                        rotation=rotation)

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
                    sucessful_img_paths.append(img_path)

        return X, sucessful_img_paths
