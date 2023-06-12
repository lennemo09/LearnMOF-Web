import re
import shutil
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import pandas as pd
from bson import ObjectId
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision
import plotly.express as px
import zipfile
import pandas as pd
import pymongo

MODEL_NAME = "LearnMOF_MODEL_BATCH5_tested_acc95"
ALLOWED_EXTENSIONS = ('jpg', 'jpeg', 'png', 'gif')

MODEL_DIR = "."
RESULTS_DIR = "results"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

SEED = 1337
IMG_SIZE = 512

BATCH_SIZE = 4  # Number of images per batch to load into memory, since we are testing only a couple of images, 1 is enough
NUM_WORKERS = 1 # Number of CPU cores to load images

WELLS_PER_ROW = 4

app = Flask(__name__, static_url_path='/static')

image_paths = []  # Global list to store the paths of uploaded images
metadata_df = None

class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize the images
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, image_path

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

    global metadata_df
    metadata_df = None
    
     # Check if the 'metadata' field is present in the request files
    if 'metadata' in request.files:
        metadata_file = request.files['metadata']

        if metadata_file.filename.endswith('.csv'):
            # Process the metadata CSV file
            metadata_df = pd.read_csv(metadata_file)

            # Clear all rows with all NaN values
            metadata_df.dropna(how='all', inplace=True)

            metadata_df['real_idx'] = metadata_df['real_idx'].astype('int')
            metadata_df['well1'] = metadata_df['well1'].astype('int')
            metadata_df['well2'] = metadata_df['well2'].astype('int')
            metadata_df['well3'] = metadata_df['well3'].astype('int')
            metadata_df['well4'] = metadata_df['well4'].astype('int')
            metadata_df['real_idx'] = metadata_df['real_idx'].astype('str')

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
                if not extracted_file_path.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                    continue

                destination_path = UPLOAD_FOLDER + '/' + os.path.basename(image_file)

                # Check if file with same name exists
                if not os.path.exists(destination_path):
                    shutil.move(image_file, UPLOAD_FOLDER)
                    image_paths.append(destination_path)
                # If filename already exists, check if same image, if not same image add a suffix
                else:
                    new_path = rename_image_with_suffix(image_file, UPLOAD_FOLDER)
                    shutil.move(image_file, new_path)
                    image_paths.append(new_path)
            
        elif file.filename.endswith('.jpg'):
            # Name clashing for jpg uploads
            file_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename

            if not os.path.exists(file_path):
                file.save(file_path)
                image_paths.append(file_path)
            else:
                new_path = rename_image_with_suffix(file_path, UPLOAD_FOLDER)
                file.save(new_path)
                image_paths.append(new_path)

    # Redirect to the first result page
    return redirect(url_for('process_images'))

@app.route('/process_images')
def process_images():
    # Load images from the UPLOADS_DIR or any other appropriate directory

    # Perform inference on the images using PyTorch model
    # Load images into a PyTorch dataset
    dataset = ImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Create empty lists to store the results
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
        
        # Store the results in the MongoDB database for each image in the batch
        for i, image_path in enumerate(image_paths_batch):
            # Create a new document for the image result
            probs = probabilities_batch[i].tolist()
            image_name = os.path.basename(image_path)

            # Check if an entry with the same image_name already exists in the database
            existing_entry = collection.find_one({'image_name': image_name})
            
            if existing_entry:
                # Update the existing entry with the new result
                collection.update_one(
                    {'_id': existing_entry['_id']},
                    {'$set': {
                        'assigned_label': class_names[predicted_classes_batch[i].item()],
                        'approved': False,
                        'probabilities': {class_names[j]: float(probs[j]) for j in range(len(probs))}
                    }}
                )
            else:
                # Create a new document for the image result
                new_db_entry = {
                    'image_name': image_name,
                    'image_path': image_path,
                    'assigned_label': class_names[predicted_classes_batch[i].item()],
                    'approved': False,
                    'probabilities': {class_names[j]: float(probs[j]) for j in range(len(probs))},
                    'linker': None,
                    'start_date_year': None,
                    'start_date_month': None,
                    'start_date_day': None,
                    'plate_index': None,
                    'image_index': None,
                    'magnification': None,
                    'reaction_time': None,
                    'temperature': None,
                    'ctot': None,
                    'loglmratio': None
                }

                # Checks if metadata is available for the image
                # If metadata is available, store the metadata in the database
                # If metadata is not available, store null values in the database
                if metadata_df is not None:
                    if re.match(r'position\d{3}.jpg', image_name):
                        image_index = int(image_name[8:11])

                        # TODO: Handle when image magninifcation is not x400
                        well_index = int((image_index - 1) // WELLS_PER_ROW) + 1
                        row_index = int(np.ceil(well_index / WELLS_PER_ROW)) - 1
                        print(f"Image index: {image_index}, well index: {well_index}, row index: {row_index}")

                        # Select row from metadata_df pandas dataframe by row_index
                        # TODO: Handle when image index not in metadata file
                        row = metadata_df.iloc[row_index]
                        # row['real_idx] is in the format 2023042426, which is year, month, day, plate_index
                        real_idx = row['real_idx']
                        year = int(real_idx[:4])
                        month = int(real_idx[4:6])
                        day = int(real_idx[6:8])
                        plate_index = int(real_idx[8:10])

                        new_db_entry['start_date_year'] = year
                        new_db_entry['start_date_month'] = month
                        new_db_entry['start_date_day'] = day
                        new_db_entry['plate_index'] = plate_index
                        new_db_entry['image_index'] = image_index

                        new_db_entry['linker'] = row['acronym']
                        new_db_entry['reaction_time'] = row['time']
                        new_db_entry['temperature'] = row['temp']
                        new_db_entry['ctot'] = row['ctot']
                        new_db_entry['loglmratio'] = row['loglmratio']

                # Insert the result document into the database collection
                entry = collection.insert_one(new_db_entry)
                entry_id = entry.inserted_id
    
    # Redirect to the 'result' page
    return redirect(url_for('result_by_index', index=0))

@app.route('/browse/database/<string:db_id>')
def result_from_db(db_id):
    result_document = collection.find_one({'_id': ObjectId(db_id)})

    # Check if the result document exists
    if result_document is None:
        return f'Result not found for db_id: {db_id}', 404
    
    file_path = result_document['image_path']
    # Extract the result information from the document
    predicted_class = result_document['assigned_label']
    probabilities = result_document['probabilities']
    probabilities_list = list(probabilities.values())

    print(predicted_class)

    print(f"Image {file_path}: Predicted class: {predicted_class} with probabilities: {probabilities}.")
    
    # Create the bar chart data
    labels = list(class_names.values())
    labels[0], labels[1] = labels[1], labels[0]
    probabilities_list[0], probabilities_list[1] = probabilities_list[1], probabilities_list[0]
    colors = ['rgb(52, 129, 237)', 'rgb(130, 232, 133)', 'rgb(224, 93, 70)']  # Customize the colors if needed

    # Drawing the stacked bar chart horizontally
    names_col = ['Class', '#','Probability']
    plotting_data = [[labels[i], 0, probabilities_list[i]] for i in range(len(labels))]
    plot_df = pd.DataFrame(data=plotting_data, columns=names_col)

    fig = px.bar(plot_df, x='Probability', y='#', color='Class' ,title='Classification probabilities', orientation='h',
                 height=100, hover_data={"Class":True,"Probability":True,"#":False},
                 color_discrete_sequence=colors)
    
    fig.update_layout(template='simple_white',margin=dict(l=0,r=0,b=0,t=0),
                     xaxis_range=[0,1], showlegend=False)
    
    # Set the y axis visibility OFF
    fig.update_yaxes(title='y', visible=False, showticklabels=False)

    # Convert the Figure object to an HTML string
    chart_html = fig.to_html(full_html=False)

    result = {
        'image_path': file_path,
        'image_name': os.path.basename(file_path),
        'db_id': str(result_document['_id']),
        'predicted_class': predicted_class,
        'approved': result_document['approved'],
        'probabilities': probabilities,
        'chart_html': chart_html,
        'magnification': result_document['magnification'] if result_document['magnification'] is not None else 'n/a',
        'start_day': result_document['start_date_day'] if result_document['start_date_day'] is not None else '?',
        'start_month': result_document['start_date_month'] if result_document['start_date_month'] is not None else '?',
        'start_year': result_document['start_date_year'] if result_document['start_date_year'] is not None else '?',
        'plate_index': result_document['plate_index'] if result_document['plate_index'] is not None else 'n/a',
        'image_index': result_document['image_index'] if result_document['image_index'] is not None else 'n/a',
        'linker': result_document['linker'] if result_document['linker'] is not None else 'n/a',
        'reaction_time': result_document['reaction_time'] if result_document['reaction_time'] is not None else 'n/a',
        'temperature': result_document['temperature'] if result_document['temperature'] is not None else 'n/a',
        'ctot': result_document['ctot'] if result_document['ctot'] is not None else 'n/a',
        'loglmratio': result_document['loglmratio'] if result_document['loglmratio'] is not None else 'n/a'
    }

    return render_template('result_db.html', result=result)

@app.route('/uploadresult/<int:index>')
def result_by_index(index):
    # TODO: Change result page from using image_paths to using database ID's.
    page_index = index % len(image_paths)

    if page_index < 0 or page_index >= len(image_paths):
        return 'Invalid index', 400

    file_path = image_paths[page_index]
    file_name = os.path.basename(file_path)

    result_document = collection.find_one({'image_name': file_name})

    # Check if the result document exists
    if result_document is None:
        return f'Result not found for {file_name}', 404

    # Extract the result information from the document
    predicted_class = result_document['assigned_label']
    probabilities = result_document['probabilities']
    probabilities_list = list(probabilities.values())

    print(predicted_class)

    print(f"Image {file_path}: Predicted class: {predicted_class} with probabilities: {probabilities}.")
    
    # Create the bar chart data
    labels = list(class_names.values())
    labels[0], labels[1] = labels[1], labels[0]
    probabilities_list[0], probabilities_list[1] = probabilities_list[1], probabilities_list[0]
    colors = ['rgb(52, 129, 237)', 'rgb(130, 232, 133)', 'rgb(224, 93, 70)']  # Customize the colors if needed

    # Drawing the stacked bar chart horizontally
    names_col = ['Class', '#','Probability']
    plotting_data = [[labels[i], 0, probabilities_list[i]] for i in range(len(labels))]
    plot_df = pd.DataFrame(data=plotting_data, columns=names_col)

    fig = px.bar(plot_df, x='Probability', y='#', color='Class' ,title='Classification probabilities', orientation='h',
                 height=100, hover_data={"Class":True,"Probability":True,"#":False},
                 color_discrete_sequence=colors)
    
    fig.update_layout(template='simple_white',margin=dict(l=0,r=0,b=0,t=0),
                     xaxis_range=[0,1], showlegend=False)
    
    # Set the y axis visibility OFF
    fig.update_yaxes(title='y', visible=False, showticklabels=False)

    # Convert the Figure object to an HTML string
    chart_html = fig.to_html(full_html=False)

    result = {
        'image_path': file_path,
        'image_name': os.path.basename(file_path),
        'db_id': str(result_document['_id']),
        'predicted_class': predicted_class,
        'approved': result_document['approved'],
        'probabilities': probabilities,
        'chart_html': chart_html,
        'uploaded_batch_index': page_index,
        'uploaded_batch_size': len(image_paths),
        'magnification': result_document['magnification'] if result_document['magnification'] is not None else 'n/a',
        'start_day': result_document['start_date_day'] if result_document['start_date_day'] is not None else '?',
        'start_month': result_document['start_date_month'] if result_document['start_date_month'] is not None else '?',
        'start_year': result_document['start_date_year'] if result_document['start_date_year'] is not None else '?',
        'plate_index': result_document['plate_index'] if result_document['plate_index'] is not None else 'n/a',
        'image_index': result_document['image_index'] if result_document['image_index'] is not None else 'n/a',
        'linker': result_document['linker'] if result_document['linker'] is not None else 'n/a',
        'reaction_time': result_document['reaction_time'] if result_document['reaction_time'] is not None else 'n/a',
        'temperature': result_document['temperature'] if result_document['temperature'] is not None else 'n/a',
        'ctot': result_document['ctot'] if result_document['ctot'] is not None else 'n/a',
        'loglmratio': result_document['loglmratio'] if result_document['loglmratio'] is not None else 'n/a'
    }

    return render_template('result_upload.html', result=result)
    
@app.route('/browse')
def browse():
    # Query the MongoDB database to retrieve all entries
    entries = collection.find()

    # Create a list to store the image data
    images = []

    # Iterate over the entries and extract the image data
    for entry in entries:
        image_data = {
            'db_id': str(entry['_id']),
            'image_path': entry.get('image_path', ''),
            'image_name': entry.get('image_name', ''),
            'approved': entry.get('approved', ''),
            'assigned_label': entry.get('assigned_label', '')
        }
        images.append(image_data)

    # Render the 'database.html' template with the image data
    return render_template('database.html', images=images)

@app.route('/<path:filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/update_approval/<string:db_id>', methods=['POST'])
def update_approval(db_id):
    if request.method == 'POST':
        approved = request.form.get('approve', False)
        label = request.form.get('label', None)

        print('Approved:', approved, 'Label:', label)

        # Update the existing entry with the new result
        collection.update_one(
            {'_id': ObjectId(db_id)},
            {'$set': {
                'assigned_label': label,
                'approved': approved}
            }
        )

    return '', 204  # Return an empty response with a 204 status code

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
    
    try:
        collection.find()
    except:
        print('MongoDB not running, exiting...')
    

    print(f"Using torchvision version {torchvision.__version__}.")
    print(f"Using torch version {torch.__version__}.")

    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU {device} for PyTorch.")
    else:
        device = torch.device("cpu")
        print("Using CPU for PyTorch.")
    

    model = torch.load(MODEL_NAME,map_location=device)
    model = model.to(device)

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
