import os
import re

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from main import collection, device, model
from main.enums import BATCH_SIZE, IMG_SIZE, NUM_WORKERS, Label

WELLS_PER_ROW = 4


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize the images
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, image_path


def perform_reference(image_paths, process_id = None, inference_progress_dict = None):
    # Load images from the UPLOADS_DIR or any other appropriate directory

    # Perform inference on the images using PyTorch model
    # Load images into a PyTorch dataset
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    predicted_classes_list = []
    probabilities_list = []

    # Iterate over the batches in the dataloader
    for batch_num, (images, image_paths_batch) in enumerate(dataloader):
        print(f"Processing batch {batch_num}/{len(dataloader)}")
        # Perform inference on the batch of images using your model
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            # Extract the predicted class and probabilities from the outputs
            # Adjust the code according to the structure of your model's output
            predicted_classes_batch = torch.argmax(outputs, dim=1)
            probabilities_batch = torch.nn.functional.softmax(outputs, dim=1)

        # Append the batch results to the lists
        predicted_classes_list.extend(predicted_classes_batch.tolist())
        probabilities_list.extend(probabilities_batch.tolist())

        if process_id is not None and inference_progress_dict is not None:
            # Update progress for this process
            inference_progress_dict[process_id] = int((batch_num + 1) / len(dataloader) * 100)
    inference_progress_dict[process_id] = 100
    
    return predicted_classes_list, probabilities_list


def update_inference_to_db(
    image_paths, predicted_classes_list: list, probabilities_list: list, metadata_df
):
    images_db_id = []
    for i, image_path in enumerate(image_paths):
        probs = probabilities_list[i]
        image_name = os.path.basename(image_path)

        existing_entry = collection.find_one({"image_name": image_name})

        if existing_entry:
            collection.update_one(
                {"_id": existing_entry["_id"]},
                {
                    "$set": {
                        "assigned_label": LABEL_MAPPING[predicted_classes_list[i]],
                        "approved": False,
                        "probabilities": {
                            LABEL_MAPPING[j]: float(probs[j]) for j in range(len(probs))
                        },
                    }
                },
            )
            images_db_id.append(str(existing_entry["_id"]))

        else:
            new_db_entry = {
                "image_name": image_name,
                "image_path": image_path,
                "assigned_label": LABEL_MAPPING[predicted_classes_list[i]],
                "approved": False,
                "probabilities": {
                    LABEL_MAPPING[j]: float(probs[j]) for j in range(len(probs))
                },
                "linker": None,
                "start_date_year": None,
                "start_date_month": None,
                "start_date_day": None,
                "plate_index": None,
                "well_index": None,
                "image_index": None,
                "magnification": None,
                "reaction_time": None,
                "temperature": None,
                "ctot": None,
                "loglmratio": None,
            }

            added_entry = collection.insert_one(new_db_entry)
            images_db_id.append(str(added_entry.inserted_id))

        if metadata_df is not None:
            update_metadata(image_name, metadata_df)

    return images_db_id


def update_metadata(image_name, metadata_df):
    if re.match(r"position(\d*).jpg", image_name):
        image_index = int(re.match(r"position(\d*).jpg", image_name).group(1))
        magnification = None
    elif re.match(r"^\d{8}-(\d*)-(\d*)x_position(\d*).jpg$", image_name):
        match = re.match(r"^(\d{4})(\d{2})(\d{2})-(\d*)-(\d*)x_position(\d*).jpg$", image_name)
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        plate_index = int(match.group(4))
        magnification = int(match.group(5))

        image_index = int(match.group(6))
        
    else:
        image_index = None

    myquery = {"image_name": image_name}

    if image_index is not None:
        well_index = int((image_index - 1) // WELLS_PER_ROW) + 1
        row_index = int(np.ceil(well_index / WELLS_PER_ROW)) - 1

        row = metadata_df.iloc[row_index]

        if row['real_idx'] != '':
            real_idx = row["real_idx"]
            year = int(real_idx[:4])
            month = int(real_idx[4:6])
            day = int(real_idx[6:8])
            plate_index = int(real_idx[8:10])

        newvalues = {
            "$set": {
                "linker": row["acronym"],
                "magnification": int(magnification),
                "reaction_time": int(row["time"]),
                "temperature": int(row["temp"]),
                "ctot": float(row["ctot"]),
                "loglmratio": float(row["loglmratio"]),
                "start_date_year": str(year),
                "start_date_month": str(month),
                "start_date_day": str(day),
                "plate_index": int(plate_index),
                "image_index": int(image_index),
                "well_index": int(well_index),
            }
        }
        collection.update_one(myquery, newvalues)


LABEL_MAPPING = {0: Label.CHALLENGING_CRYSTAL, 1: Label.CRYSTAL, 2: Label.NON_CRYSTAL}
