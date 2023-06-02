import shutil
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import plotly.express as px
import zipfile
import pandas as pd
import pymongo

MODEL_NAME = "LearnMOF_MODEL_BATCH5_tested_acc95"
ALLOWED_EXTENSIONS = ('jpg', 'jpeg', 'png', 'gif')

MODEL_DIR = "."
RESULTS_DIR = "results"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

# IMAGES_DIR = Path("images")

SEED = 1337
IMG_SIZE = 512

BATCH_SIZE = 1  # Number of images per batch to load into memory, since we are testing only a couple of images, 1 is enough
NUM_WORKERS = 1 # Number of CPU cores to load images

app = Flask(__name__, static_url_path='/static')

image_paths = []  # Global list to store the paths of uploaded images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global image_paths
    image_paths = []

     # Check if the 'images' field is present in the request files
    if 'images' not in request.files:
        return 'No images uploaded', 400

    images = request.files.getlist('images')  # Get a list of all uploaded image files

    # Check if any files were selected
    if not images:
        return 'No images selected', 400

    metadata_file = None
     # Check if the 'metadata' field is present in the request files
    if 'metadata' in request.files:
        metadata_file = request.files['metadata']
        print(metadata_file)
        if metadata_file.filename.endswith('.csv'):
            # Process the metadata CSV file
            metadata_df = pd.read_csv(metadata_file)
        # else:
        #     return 'Metadata should be in CSV format', 400

    for file in images:
        # Check if the file has a filename
        if file.filename == '':
            return 'One or more files have no filename', 400

        # Check if the file is a zip file
        if file.filename.endswith('.zip'):
            # Save the zip file to the upload folder
            file_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
            file.save(file_path)

            # Extract the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall('temp')

            # Remove the zip file
            os.remove(file_path)

            # Check the extracted files for directories and non-image files
            extracted_files = os.listdir('temp')
            new_image_files = []
            for extracted_file in extracted_files:
                extracted_file_path = 'temp' + '/' + extracted_file

                if os.path.isdir(extracted_file_path) or not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                    # Raise an error if a directory is found
                    os.remove(extracted_file_path)
                    return 'Zip file contains directories', 400
                
                if extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                    new_image_files.append(extracted_file_path)

            # Raise an error if no image files are found
            if not new_image_files:
                return 'Zip file does not contain any images', 400

            # Move the image files to the UPLOAD_FOLDER directory
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            for image_file in new_image_files:
                # Check if file with same name exists
                if not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                    continue

                destination_path = UPLOAD_FOLDER + '/' + os.path.basename(image_file)

                if not os.path.exists(destination_path):
                    shutil.move(image_file, UPLOAD_FOLDER)
                    image_paths.append(destination_path)
                # If filename already exists, check if same image, if not same image add a suffix
                else:
                    new_path = rename_image_with_suffix(image_file, UPLOAD_FOLDER)
                    shutil.move(image_file, new_path)
                    image_paths.append(new_path)
            
        elif file.filename.endswith('.jpg'):
            file_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
            file.save(file_path)
            image_paths.append(file_path)

    # Redirect to the first result page
    return redirect(url_for('result', index=0))


@app.route('/result/<int:index>')
def result(index):
    index = int(request.args.get('index', 0))

    if index < 0 or index >= len(image_paths):
        return 'Invalid index', 400

    file_path = image_paths[index]

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
    #   predicted_probabilities: [float,float,float]|null;
    #   assigned_label: string;
    #   label_approved: boolean;
    #   linker: string|null;
    #   startdate : {year:int,month:int,day:int}|null;
    #   plate_index : int|null;
    #   image_index : int|null;
    #   magnification: int|null;
    #   conditions: {
    #       time: int|null;
    #       temp: float|null;
    #       ctot: float|null;
    #       loglmratio: float|null;
    #   }
    # }
    result = {
        'image_path': file_path,
        'predicted_class': predicted_class,
        'probabilities': {class_names[i]: prediction[i] for i in range(len(prediction))},
        'chart_html': chart_html,
        'index': index
    }

    new_db_entry = {
        'image_name' : os.path.basename(file_path),
        'image_path': file_path,
        'assigned-label': predicted_class,
        'approved' : False,
        'probabilities': {class_names[i]: float(prediction[i]) for i in range(len(prediction))},
    }

    # print(new_db_entry)

    # insert_result = collection.insert_one(new_db_entry)

    return render_template('result.html', result=result)

@app.route('/<path:filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def is_same_image(image_path1, image_path2):
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    if image1.size != image2.size:
        return False

    pixels1 = image1.load()
    pixels2 = image2.load()

    width, height = image1.size
    for x in range(width):
        for y in range(height):
            if pixels1[x, y] != pixels2[x, y]:
                return False

    return True

def rename_image_with_suffix(image_file, destination_dir):
    filename = os.path.basename(image_file)
    destination_path = destination_dir + '/' + filename

    if os.path.exists(destination_path):
        # Check if the existing image is the same as the one being moved
        if not is_same_image(image_file, destination_path):
            # Generate a new filename with a suffix
            suffix = 1
            while True:
                new_filename = f"{os.path.splitext(filename)[0]}_{suffix}{os.path.splitext(filename)[1]}"
                print(f'Image with name exists {filename} with different content, renaming to {new_filename}.')
                new_destination_path = destination_dir + '/' + new_filename
                if not os.path.exists(new_destination_path):
                    break
                suffix += 1

            destination_path = new_destination_path

    # Move the image to the destination directory
    return destination_path
    # shutil.move(image_file, destination_path)

if __name__ == '__main__':
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['learnmof']
    collection = db['data']

    print(db)

    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU {device} for PyTorch.")
    else:
        device = torch.device("cpu")
        print("Using CPU for PyTorch.")
    

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
    UPLOAD_FOLDER = 'static/images'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(host="0.0.0.0")
