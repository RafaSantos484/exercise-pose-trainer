import argparse
import sys
import os
import uuid


def rename_files_random(path: str) -> None:
    if not os.path.isdir(path):
        print('Invalid folder path')
        return

    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):
            _, extension = os.path.splitext(file)
            new_name = f'{uuid.uuid4().hex}{extension}'
            while os.path.exists(os.path.join(path, new_name)):
                new_name = f'{uuid.uuid4().hex}{extension}'
            new_path = os.path.join(path, new_name)

            os.rename(filepath, new_path)
            print(f'{file} -> {new_name}')


def rename_files_sequential(folder: str):
    if not os.path.isdir(folder):
        print('Invalid folder path')
        return

    files = [f for f in os.listdir(
        folder) if os.path.isfile(os.path.join(folder, f))]
    files.sort()

    total_files = len(files)
    # Define o número de dígitos com base na quantidade de arquivos
    padding = len(str(total_files))

    for i, file in enumerate(files, start=1):
        filepath = os.path.join(folder, file)
        _, extension = os.path.splitext(file)
        new_name = f'{str(i).zfill(padding)}{extension}'
        new_path = os.path.join(folder, new_name)

        os.rename(filepath, new_path)
        print(f'{file} -> {new_name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the folder')
    args = parser.parse_args()
    rename_files_sequential(args.path)
