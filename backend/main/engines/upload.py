import os
import shutil
import zipfile

import pandas as pd
from PIL import Image

from main import app
from main.enums import ALLOWED_EXTENSIONS, UPLOAD_FOLDER


def get_metadata_from_csv(metadata_file):
    metadata_df = pd.read_csv(metadata_file)

    # Clear all rows with all NaN values
    metadata_df.dropna(how="all", inplace=True)

    metadata_df["real_idx"] = metadata_df["real_idx"].astype("int")
    metadata_df["well1"] = metadata_df["well1"].astype("int")
    metadata_df["well2"] = metadata_df["well2"].astype("int")
    metadata_df["well3"] = metadata_df["well3"].astype("int")
    metadata_df["well4"] = metadata_df["well4"].astype("int")
    metadata_df["real_idx"] = metadata_df["real_idx"].astype("str")

    return metadata_df


def handle_jpg_file(file, file_path):
    if not os.path.exists(file_path):
        file.save(file_path)
        return file_path
    else:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = f"{temp_dir}/{os.path.basename(file_path)}"
        file.save(file_path)
        new_path = rename_image_with_suffix(file_path, UPLOAD_FOLDER)
        shutil.move(file_path, new_path)
        return new_path


def handle_zip_file(file, file_path):
    # Save the zip file to the upload folder
    file.save(file_path)

    # Extract the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall("temp")

    # Remove the zip file
    os.remove(file_path)

    # Check the extracted files for directories and non-image files
    extracted_files = os.listdir("temp")
    new_image_files = []
    for extracted_file in extracted_files:
        extracted_file_path = "temp" + "/" + extracted_file

        if os.path.isdir(
            extracted_file_path
        ) or not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            # Raise an error if a directory is found
            os.remove(extracted_file_path)
            return "Zip file contains directories", 400

        if extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            new_image_files.append(extracted_file_path)

    # Raise an error if no image files are found
    if not new_image_files:
        return "Zip file does not contain any images", 400

    # Move the image files to the UPLOAD_FOLDER directory
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    for image_file in new_image_files:
        # if not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        #     continue

        destination_path = (
            app.config["UPLOAD_FOLDER"] + "/" + os.path.basename(image_file)
        )

        # Check if file with same name exists
        if os.path.exists(destination_path):
            destination_path = rename_image_with_suffix(
                image_file, app.config["UPLOAD_FOLDER"]
            )

        shutil.move(image_file, destination_path)


def rename_image_with_suffix(image_file, destination_dir):
    filename = os.path.basename(image_file)
    destination_path = destination_dir + "/" + filename

    if os.path.exists(destination_path):
        # Check if the existing image is the same as the one being moved
        if not is_same_image(image_file, destination_path):
            # Generate a new filename with a suffix
            suffix = 1
            while True:
                new_filename = f"{os.path.splitext(filename)[0]}_{suffix}{os.path.splitext(filename)[1]}"
                print(
                    f"Image with name exists {filename} with different content, renaming to {new_filename}."
                )
                new_destination_path = destination_dir + "/" + new_filename
                if not os.path.exists(new_destination_path):
                    break
                suffix += 1

            destination_path = new_destination_path

    return destination_path


def is_same_image(image_path_1, image_path_2):
    image_1 = Image.open(image_path_1)
    image_2 = Image.open(image_path_2)

    if image_1.size != image_2.size:
        return False

    pixels_1 = image_1.load()
    pixels_2 = image_2.load()

    width, height = image_1.size
    for x in range(width):
        for y in range(height):
            if pixels_1[x, y] != pixels_2[x, y]:
                return False

    return True
