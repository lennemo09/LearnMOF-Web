from flask import Flask, request, render_template
import os
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

MODEL_NAME = "LearnMOF_MODEL_BATCH5_tested_acc95"

MODEL_DIR = "."
RESULTS_DIR = "results"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

# IMAGES_DIR = Path("images")

SEED = 1337
IMG_SIZE = 512

BATCH_SIZE = 1  # Number of images per batch to load into memory, since we are testing only a couple of images, 1 is enough
NUM_WORKERS = 1 # Number of CPU cores to load images

# Setup device-agnostic code
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(MODEL_NAME,map_location=device)
model = model.to(device)

test_transforms_list = [
    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the images
] 

test_transform = transforms.Compose(test_transforms_list)

# Define the class names and the target directories
class_names = {
    0: 'challenging-crystal',
    1: 'crystal',
    2: 'non-crystal'
}

model.eval()


app = Flask(__name__, static_url_path='/static')

# Set the upload folder
UPLOAD_FOLDER = MODEL_DIR + "/" + "static" + "/" + "images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was submitted
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    # Check if the file has a filename
    if file.filename == '':
        return 'No file selected', 400

    # Save the file to the upload folder
    file_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
    file.save(file_path)

    # Load and preprocess the image
    img = Image.open(file_path)

    tensor_img = test_transform(img).unsqueeze(0)
    tensor_img = tensor_img.to(device)
    prediction = torch.nn.functional.softmax(model(tensor_img),dim=1).cpu().detach().numpy().flatten()

    print(prediction)

    predicted_class = class_names[prediction.argmax()]

    print(predicted_class)

    print(f"Image {file_path}: Predicted class: {class_names[prediction.argmax()]} with probabilities: {prediction}.")

    result = {
        'image_path': file_path,
        'predicted_class': predicted_class,
        'probabilities': {class_names[i]: prediction[i] for i in range(3)}
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run()
