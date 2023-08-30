import os
import uuid

from bson import ObjectId
from flask import jsonify, request, send_from_directory

from main import app, collection
from main.commons.decorators import parse_args_with
from main.engines.inference import perform_reference, update_inference_to_db
from main.engines.upload import get_metadata_from_csv, handle_jpg_file, handle_zip_file
from main.enums import UPLOAD_FOLDER
from main.schemas.browse import GetFilteredImagesSchema, UpdateApproval
from main.schemas.inference import UpdateProcessImage

metadata_df = None
image_ids_list = []

# Add a dictionary to store progress for different processes
inference_progress = {}
prepare_data_progress = {}

@app.get("/inference_progress/<process_id>")
def get_inference_progress(process_id):
    return jsonify({"progress": inference_progress.get(process_id, 0)})

@app.get("/prepare_data_progress/<process_id>")
def get_prepare_data_progress(process_id):
    return jsonify({"progress": prepare_data_progress.get(process_id, 0)})

@app.get("/get_inference_process_id")
def get_inference_process_id():
    process_id = str(uuid.uuid4())  # Generate a unique ID for this inference process
    return jsonify({"process_id": process_id})


@app.route("/upload", methods=["POST"])
def upload():
    images = request.files.getlist("images")
    metadata_file = request.files.get("metadata")
    process_id = request.form.get('process_id')

    print("Received formData", request)

    print("Uploading images with process id:", process_id)

    if not images and not metadata_file:
        return "No files selected", 400

    global metadata_df
    metadata_df = None

    if metadata_file:
        metadata_df = get_metadata_from_csv(metadata_file)

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

    return jsonify({"image_paths": image_paths})


@app.post("/process_images")
@parse_args_with(UpdateProcessImage)
def process_images(args: UpdateProcessImage):
    inference_progress[args.process_id] = 0  # Initialize progress
    predicted_classes_list, probabilities_list = perform_reference(args.image_paths, args.process_id, inference_progress)
    image_ids = update_inference_to_db(
        args.image_paths, predicted_classes_list, probabilities_list, metadata_df
    )
    print(f"predicted_classes_list {predicted_classes_list}")
    print(f"probabilities_list: {probabilities_list}")

    # Remove progress entry once the inference is complete
    del inference_progress[args.process_id]

    # After starting the inference process, return the process ID
    return jsonify({"image_ids": image_ids, "process_id": args.process_id})


@app.route("/update_approval/<string:db_id>", methods=["POST"])
@parse_args_with(UpdateApproval)
def update_approval(db_id, args: UpdateApproval, **__):
    if request.method == "POST":
        update_approval = UpdateApproval(approved=args.approved, label=args.label)

        print("Approved:", update_approval.approved, "Label:", update_approval.label)

        # Update the existing entry with the new result
        collection.update_one(
            {"_id": ObjectId(db_id)},
            {
                "$set": {
                    "assigned_label": update_approval.label,
                    "approved": update_approval.approved,
                }
            },
        )

    return "", 204  # Return an empty response with a 204 status code


@app.route("/<path:filename>")
def uploaded_image(filename):
    directory = os.path.abspath(os.path.join(app.root_path, os.pardir))
    print(f"file_name {filename}")
    return send_from_directory(directory, filename)


@app.get("/browse")
@parse_args_with(GetFilteredImagesSchema)
def browse(args: GetFilteredImagesSchema):
    if args.image_ids:
        image_ids = [ObjectId(id) for id in args.image_ids]
        entries = collection.find({"_id": {"$in": image_ids}})
    else:
        entries = collection.find(args.dict(exclude_none=True))

    images = []
    for entry in entries:
        image_data = {
            "db_id": str(entry["_id"]),
            "image_path": entry.get("image_path", ""),
            "image_name": entry.get("image_name", ""),
            "approved": entry.get("approved", ""),
            "assigned_label": entry.get("assigned_label", ""),
            "linker": entry.get("linker", ""),
            "temperature": entry.get("temperature", None),
            "reaction_time": entry.get("reaction_time", None)
        }
        images.append(image_data)


    global image_ids_list
    image_ids = collection.find({}, {"_id": 1})
    image_ids_list = [str(image_id["_id"]) for image_id in image_ids]  # Convert ObjectId to str

    return jsonify(images)


@app.route("/browse/<string:db_id>")
def result_from_db(db_id):
    result_document = collection.find_one({"_id": ObjectId(db_id)})

    if result_document is None:
        return f"Result not found for db_id: {db_id}", 404

    file_path = result_document["image_path"]

    predicted_class = result_document["assigned_label"]
    probabilities = result_document["probabilities"]

    print(
        f"Image {file_path}: Predicted class: {predicted_class} with probabilities: {probabilities}."
    )

    result = {
        "image_path": file_path,
        "image_name": os.path.basename(file_path),
        "db_id": str(result_document["_id"]),
        "predicted_class": predicted_class,
        "approved": result_document["approved"],
        "probabilities": probabilities,
        "magnification": result_document["magnification"]
        if result_document["magnification"] is not None
        else "n/a",
        "start_day": result_document["start_date_day"]
        if result_document["start_date_day"] is not None
        else "?",
        "start_month": result_document["start_date_month"]
        if result_document["start_date_month"] is not None
        else "?",
        "start_year": result_document["start_date_year"]
        if result_document["start_date_year"] is not None
        else "?",
        "plate_index": result_document["plate_index"]
        if result_document["plate_index"] is not None
        else "n/a",
        "image_index": result_document["image_index"]
        if result_document["image_index"] is not None
        else "n/a",
        "well_index": result_document["well_index"]
        if result_document["well_index"] is not None
        else "n/a",
        "linker": result_document["linker"]
        if result_document["linker"] is not None
        else "n/a",
        "reaction_time": result_document["reaction_time"]
        if result_document["reaction_time"] is not None
        else "n/a",
        "temperature": result_document["temperature"]
        if result_document["temperature"] is not None
        else "n/a",
        "ctot": result_document["ctot"]
        if result_document["ctot"] is not None
        else "n/a",
        "loglmratio": result_document["loglmratio"]
        if result_document["loglmratio"] is not None
        else "n/a",
    }

    return jsonify(result)

@app.route('/all_image_ids')
def get_all_image_ids():
    return jsonify(image_ids_list)

@app.route('/filter_unique_values')
def get_unique_values():

    pipeline = [
        {"$group": {
            "_id": None,
            "labelOptions": {"$addToSet": "$assigned_label"},
            "statusOptions": {"$addToSet": "$approved"},
            "linkerOptions": {"$addToSet": "$linker"},
            "magnificationOptions": {"$addToSet": "$magnification"},
            "startYearOptions": {"$addToSet": "$start_date_year"},
            "startMonthOptions": {"$addToSet": "$start_date_month"},
            "startDayOptions": {"$addToSet": "$start_date_day"},
            "reactionTimeOptions": {"$addToSet": "$reaction_time"},
            "temperatureOptions": {"$addToSet": "$temperature"}
        }},
        {"$project": {
            "_id": 0,
            "labelOptions": 1,
            "statusOptions": 1,
            "linkerOptions": 1,
            "magnificationOptions": 1,
            "startYearOptions": 1,
            "startMonthOptions": 1,
            "startDayOptions": 1,
            "reactionTimeOptions": 1,
            "temperatureOptions": 1
        }}
    ]
    
    distinct_values = list(collection.aggregate(pipeline))
    if distinct_values:
        return jsonify(distinct_values[0])
    else:
        return jsonify({})