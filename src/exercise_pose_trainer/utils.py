import os

"""
Checks if a given file path corresponds to an image file based on its extension.
"""
def is_img_file(path: str) -> bool:
    image_extensions = ['.jpg', '.jpeg', '.png',
                        '.gif', '.bmp', '.tiff', '.webp']
    return any(path.lower().endswith(ext) for ext in image_extensions)

"""
Extracts the base name (the last component) from a given file path.
"""
def get_basename(path: str) -> str:
    return os.path.basename(os.path.normpath(path))
