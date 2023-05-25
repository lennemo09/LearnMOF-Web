from flask import Flask, request, render_template
import os
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import plotly.express as px

MODEL_NAME = "LearnMOF_MODEL_BATCH5_tested_acc95"

MODEL_DIR = "."
RESULTS_DIR = "results"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

# IMAGES_DIR = Path("images")

SEED = 1337
IMG_SIZE = 512

BATCH_SIZE = 1  # Number of images per batch to load into memory, since we are testing only a couple of images, 1 is enough
NUM_WORKERS = 1 # Number of CPU cores to load images

app = Flask(__name__, static_url_path='/static')

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
    
    # Create the bar chart data
    labels = list(class_names.values())
    labels[0], labels[1] = labels[1], labels[0]
    probabilities = prediction.tolist()
    probabilities[0], probabilities[1] = probabilities[1], probabilities[0]
    colors = ['rgb(40, 20, 255)', 'rgb(130, 232, 133)', 'rgb(209, 65, 65)']  # Customize the colors if needed

    # Drawing the stacked bar chart horizontally
    names_col = ['Class', '#','Probability']
    plotting_data = [[labels[i], 0, probabilities[i]] for i in range(len(labels))]
    plot_df = pd.DataFrame(data=plotting_data,columns=names_col)

    fig = px.bar(plot_df, x='Probability', y='#', color='Class' ,title='Classification probabilities', orientation='h',
                 height=100, hover_data={"Class":True,"Probability":True,"#":False},
                 color_discrete_sequence=colors)
    
    fig.update_layout(template='simple_white',margin=dict(l=0,r=0,b=0,t=0),
                     xaxis_range=[0,1], showlegend=False)
    
    # Set the y axis visibility OFF
    fig.update_yaxes(title='y', visible=False, showticklabels=False)

    # Convert the Figure object to an HTML string
    chart_html = fig.to_html(full_html=False)

    # MongoDB Document format:
    # {
    #   file_path : string;
    #   image_name : string;
    #   upload_date : string;
    #   predicted_probabilities: [float,float,float]|null;
    #   assigned_label: string;
    #   label_approved: boolean;
    #   linker: string|null;
    #   magnification: int|null;
    #   conditions: {
    #       time: int|null;
    #       temp: float|null;
    #       ctot: float|null;
    #       lmlogratio: float|null;
    #   }
    # }
    result = {
        'image_path': file_path,
        'predicted_class': predicted_class,
        'probabilities': {class_names[i]: prediction[i] for i in range(len(prediction))},
        'chart_html': chart_html
    }

    new_db_entry = {
        'image_name' : os.path.basename(file_path),
        'image_path': file_path,
        'assigned-label': predicted_class,
        'approved' : False,
        'probabilities': {class_names[i]: prediction[i] for i in range(len(prediction))},
    }

    print(new_db_entry)

    return render_template('result.html', result=result)

if __name__ == '__main__':
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

    # Set the upload folder
    UPLOAD_FOLDER = MODEL_DIR + "/" + "static" + "/" + "images"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run()
