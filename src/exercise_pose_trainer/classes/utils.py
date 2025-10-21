import os

import cv2
from cv2.typing import MatLike


class Utils:
    @staticmethod
    def is_img_file(path: str) -> bool:
        image_extensions = ['.jpg', '.jpeg', '.png',
                            '.gif', '.bmp', '.tiff', '.webp']
        return os.path.isfile(path) and any(path.lower().endswith(ext) for ext in image_extensions)

    @staticmethod
    def get_basename(path: str) -> str:
        return os.path.basename(os.path.normpath(path))

    @staticmethod
    def rotate_img(img: MatLike, angle: float) -> MatLike:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

    @staticmethod
    def scale_img(img: MatLike, scale_factor: float) -> MatLike:
        (h, w) = img.shape[:2]
        new_dimensions = (int(w * scale_factor), int(h * scale_factor))
        scaled_img = cv2.resize(img, new_dimensions,
                                interpolation=cv2.INTER_AREA)
        return scaled_img
