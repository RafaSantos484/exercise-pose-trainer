import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

from exercise_pose_trainer.classes.point3d import Point3d

landmarks_dict: dict[str, int] = {
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
    def get_points(cls, image_path: str, mirror=False) -> list[Point3d] | None:
        image = cv2.imread(image_path)
        if mirror:
            image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = cls._model.detect(mp_image)
        if not result.pose_world_landmarks:
            print(f'No landmarks found for {image_path}')
            return None

        landmarks = result.pose_world_landmarks[0]
        points = [Point3d.from_landmark(lm) for lm in landmarks]
        return points
