import os

from main.enums import UPLOAD_FOLDER


def get_image_paths():
    files = []
    for file_name in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, file_name)
        if os.path.isfile(file_path):
            files.append(file_path)
    print(files)
    return files
