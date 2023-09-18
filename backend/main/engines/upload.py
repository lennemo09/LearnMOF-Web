import os
import re
import shutil
import zipfile

import pandas as pd
from PIL import Image

from main import app
from main.enums import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from main.engines.inference import perform_reference, update_inference_to_db

ZIP_FILE_PATTERN = r"^\d{8}-(\d*)-(\d*)x$"

def handle_uploaded_files(images, image_paths, process_id, metadata_df, inference_progress, prepare_data_progress):
    image_paths = []
    for file_num, file in enumerate(images):
        # Check if the file has a filename
        if file.filename == "":
            return "One or more files have no filename", 400

        file_path = UPLOAD_FOLDER + "/" + file.filename
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Check if the file is a zip file
        if file.filename.endswith(".zip"):
            # Save the zip file to the upload folder
            new_paths = handle_zip_file(file, file_path, process_id, prepare_data_progress)
            if new_paths:
                image_paths.extend(new_paths)

        elif file.filename.endswith(".jpg"):
            # Name clashing for jpg uploads
            image_path = handle_jpg_file(file, file_path)
            image_paths.append(image_path)

            # Update progress for this process
            # TODO: Handle when the upload contains both ZIP and JPG files
            prepare_data_progress[process_id] = (file_num + 1) / len(images) * 100

    # Remove progress entry once the inference is complete
    del prepare_data_progress[process_id]

    process_images(process_id, image_paths, metadata_df, inference_progress)

def process_images(process_id, image_paths, metadata_df, inference_progress):
    inference_progress[process_id] = 0  # Initialize progress
    predicted_classes_list, probabilities_list = perform_reference(image_paths, process_id, inference_progress)

    image_ids = update_inference_to_db(image_paths, predicted_classes_list, probabilities_list, metadata_df)

    print(f"predicted_classes_list {predicted_classes_list}")
    print(f"probabilities_list: {probabilities_list}")

    # Remove progress entry once the inference is complete
    del inference_progress[process_id]

def get_metadata_from_csv(metadata_file):
    metadata_df = pd.read_csv(metadata_file)

    # Clear all rows with all NaN values
    metadata_df.dropna(how="all", inplace=True)

    # metadata_df["real_idx"] = metadata_df["real_idx"].astype("int")
    metadata_df["well1"] = metadata_df["well1"].astype("int") if metadata_df["well1"] is not None else None
    metadata_df["well2"] = metadata_df["well2"].astype("int") if metadata_df["well1"] is not None else None
    metadata_df["well3"] = metadata_df["well3"].astype("int") if metadata_df["well1"] is not None else None
    metadata_df["well4"] = metadata_df["well4"].astype("int") if metadata_df["well1"] is not None else None
    metadata_df["real_idx"] = metadata_df["real_idx"].astype("str") if metadata_df["well1"] is not None else ""

    return metadata_df


def handle_jpg_file(file, file_path):
    if not os.path.exists(file_path):
        file.save(file_path)
        return file_path
    else:
        temp_dir = "temp/temp_jpg"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = f"{temp_dir}/{os.path.basename(file_path)}"
        file.save(file_path)
        new_path = rename_image_with_suffix(file_path, UPLOAD_FOLDER)
        shutil.move(file_path, new_path)
        return new_path


def handle_zip_file(file, file_path, process_id=None, prepare_data_progress=None):
    # Save the zip file to the upload folder
    file.save(file_path)

    # Get the base filename from the file path
    filename = os.path.basename(file_path)

    if process_id is not None:
        temp_dir = f"temp/{process_id}"
    else:
        temp_dir = "temp"

    # Get the filename without the extension
    filename_without_extension = os.path.splitext(filename)[0]
    print(f"Extracting Zip file: {filename_without_extension}")
    # Extract the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Check if the filename matches the pattern
    match = re.match(ZIP_FILE_PATTERN, filename_without_extension)

    # Remove the zip file
    os.remove(file_path)

    image_paths = []

    # Recursively search for image files in the extracted directory
    def search_for_images(directory):
        new_image_files = []
        for root, dirs, files in os.walk(directory):
            for extracted_file in files:
                extracted_file_path = os.path.join(root, extracted_file)

                if not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                    continue

                new_image_files.append(extracted_file_path)

        return new_image_files

    extracted_files = os.listdir(temp_dir)
    found_image_files = search_for_images(temp_dir)

    # Raise an error if no image files are found
    if not found_image_files:
        return "Zip file does not contain any images", 400

    print("Moving files to upload folder")
    # Move the image files to the UPLOAD_FOLDER directory
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    for file_num, image_file in enumerate(found_image_files):
        print(image_file, process_id, prepare_data_progress)
        # if not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        #     continue
        # print(image_file)
        if not match:
            new_filename = os.path.basename(image_file)
        else:
            image_filename = os.path.basename(image_file)
            image_file_without_extension, image_file_extension = (
                os.path.splitext(image_filename)[0],
                os.path.splitext(image_filename)[1],
            )
            new_filename = f"{filename_without_extension}_{image_file_without_extension}{image_file_extension}"
            # print(f"DIRNAME: {os.path.dirname(image_file)}")
            new_image_file = f"{os.path.dirname(image_file)}/{new_filename}"
            os.rename(image_file, new_image_file)
            image_file = new_image_file

        destination_path = f"{app.config['UPLOAD_FOLDER']}/{new_filename}"

        # Check if file with same name exists
        if os.path.exists(destination_path):
            destination_path = rename_image_with_suffix(
                image_file, app.config["UPLOAD_FOLDER"]
            )
        image_paths.append(destination_path)
        shutil.move(image_file, destination_path)

        if process_id is not None and prepare_data_progress is not None:
            # Update progress for this process
            print(f"Updating data prep progress: {(file_num + 1) / len(found_image_files) * 100}")
            prepare_data_progress[process_id] = (file_num + 1) / len(found_image_files) * 100

    # Remove the zip file
    shutil.rmtree(temp_dir)
    return image_paths


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
