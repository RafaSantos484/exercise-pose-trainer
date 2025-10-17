import os


class Utils:
    @staticmethod
    def is_img_file(path: str) -> bool:
        image_extensions = ['.jpg', '.jpeg', '.png',
                            '.gif', '.bmp', '.tiff', '.webp']
        return any(path.lower().endswith(ext) for ext in image_extensions)

    @staticmethod
    def get_basename(path: str) -> str:
        return os.path.basename(os.path.normpath(path))
