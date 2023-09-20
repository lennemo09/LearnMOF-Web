import os
from PIL import Image
import pymongo

# Input and output directories
input_dir = 'static/images'
output_dir = 'static/images_thumbnails'

mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection URL
database_name = "learnmof"
collection_name = "data"

# Connect to the database and collection
db = mongo_client[database_name]
collection = db[collection_name]

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List all files in the input directory
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Resize and save images
for image_file in image_files:
    try:
        # Open the original image
        input_path = os.path.join(input_dir, image_file)
        image = Image.open(input_path)

        # Calculate the new size (25% of the original size)
        width, height = image.size
        new_width = int(width * 0.25)
        new_height = int(height * 0.25)

        # Resize the image
        smaller_image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # Generate the output filename with _small suffix
        output_file = os.path.splitext(image_file)[0] + '_small' + os.path.splitext(image_file)[1]
        output_path = os.path.join(output_dir, output_file)

        # Save the smaller image
        smaller_image.save(output_path)

        print(f'{image_file} resized and saved as {output_file}')

    except Exception as e:
        print(f'Error processing {image_file}: {str(e)}')

# Iterate through all documents in the collection
for document in collection.find():
    try:
        # Get the original image_path from the document
        image_path = document.get("image_path")

        # Generate the thumbnail_path with "_small" suffix
        thumbnail_path = image_path.replace("static/images/", "static/images_thumbnails/").replace(".jpg", "_small.jpg")

        # Update the document with the new thumbnail_path
        collection.update_one({"_id": document["_id"]}, {"$set": {"thumbnail_path": thumbnail_path}})

        print(f'Updated document with _id: {document["_id"]}')

    except Exception as e:
        print(f'Error updating document: {str(e)}')

# Close the MongoDB connection
mongo_client.close()