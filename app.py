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
    probabilities = prediction.tolist()
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']  # Customize the colors if needed

    names_col = ['Class', '#','Percentage']
    dat = [[labels[i], 0, probabilities[i]] for i in range(len(labels))]
    plot_df = pd.DataFrame(data=dat,columns=names_col)

    fig = px.bar(plot_df, x='Percentage', y='#', color='Class' ,title='Classification probabilities', orientation='h',
                 height=100, hover_data={"Class":True,"Percentage":True,"#":False})
    
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),
                     xaxis_range=[0,1])
    
    # Set the y axis visibility OFF
    fig.update_yaxes(title='y', visible=False, showticklabels=False)

    # 
    # data = [
    #     go.Bar(
    #         x=["probabilities"],
    #         y=[probabilities[0]],
    #         orientation='h',
    #         marker=dict(color=colors)
    #     ),

    #     go.Bar(
    #         x=["probabilities"],
    #         y=[probabilities[1]],
    #         orientation='h',
    #         marker=dict(color=colors)
    #     ),

    #     go.Bar(
    #         x=["probabilities"],
    #         y=[probabilities[2]],
    #         orientation='h',
    #         marker=dict(color=colors)
    #     )
    # ]

    # # Define the layout of the chart
    # layout = go.Layout(
    #     title='Class Probabilities',
    #     xaxis=dict(title='Probability'),
    #     yaxis=dict(title='Class'),
    #     hovermode='closest'
    # )

    # # Create the Figure object
    # fig = go.Figure(data=data, layout=layout)
    # fig.update_layout(barmode='stack')

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
        'probabilities': {class_names[i]: prediction[i] for i in range(3)},
        'chart_html': chart_html
    }

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
