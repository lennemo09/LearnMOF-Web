import os

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from main import collection, device, model
from main.enums import BATCH_SIZE, IMG_SIZE, NUM_WORKERS, Label


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


def perform_reference(image_paths):
    # Load images from the UPLOADS_DIR or any other appropriate directory

    # Perform inference on the images using PyTorch model
    # Load images into a PyTorch dataset
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    predicted_classes_list = []
    probabilities_list = []

    batch_count = 0
    # Iterate over the batches in the dataloader
    for images, image_paths_batch in dataloader:
        print(f"Processing batch {batch_count}/{len(dataloader)}")
        batch_count += 1
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
        update_metadata(metadata_df)

    return images_db_id


def update_metadata(metadata_df):
    df_long = pd.melt(
        metadata_df,
        id_vars=ID_VARS,
        value_vars=VALUE_VARS,
        var_name="well",
        value_name="value",
    )
    df = df_long.drop(columns="well").rename(columns={"value": "well"})

    for index, row in df.iterrows():
        position = str(row["well"]).zfill(3)
        image_name = f"position{position}.jpg"

        myquery = {"image_name": image_name}
        image_index = int(image_name[8:11])

        real_idx = row["real_idx"]
        year = int(real_idx[:4])
        month = int(real_idx[4:6])
        day = int(real_idx[6:8])
        plate_index = int(real_idx[8:10])

        newvalues = {
            "$set": {
                "linker": row["acronym"],
                "reaction_time": row["time"],
                "temperature": row["temp"],
                "ctot": row["ctot"],
                "loglmratio": row["loglmratio"],
                "start_date_year": year,
                "start_date_month": month,
                "start_date_day": day,
                "plate_index": plate_index,
                "image_index": image_index,
            }
        }

        collection.update_one(myquery, newvalues)


LABEL_MAPPING = {0: Label.CHALLENGING_CRYSTAL, 1: Label.CRYSTAL, 2: Label.NON_CRYSTAL}
ID_VARS = ["temp", "time", "linker_idx", "acronym", "ctot", "loglmratio", "real_idx"]
VALUE_VARS = ["well1", "well2", "well3", "well4"]
