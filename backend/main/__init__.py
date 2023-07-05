import os

import pymongo
import torch
from flask import Flask
from flask_cors import CORS

from main.enums import MODEL_NAME

static_url_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "static")
)
app = Flask(__name__, static_url_path=static_url_path)
app.config["UPLOAD_FOLDER"] = "static/images"

CORS(app)

mongodb_url = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/")
client = pymongo.MongoClient(mongodb_url)
db = client["learnmof"]
collection = db["data"]

# Setup device-agnostic code
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU {device} for PyTorch.")
else:
    device = torch.device("cpu")
    print("Using CPU for PyTorch.")

model = torch.load(MODEL_NAME, map_location=device)
model = model.to(device)


def register_subpackages():
    import main.controllers


register_subpackages()
