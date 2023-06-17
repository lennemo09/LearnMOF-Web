import os

import pandas as pd
import plotly.express as px
from bson import ObjectId
from flask import redirect, render_template, request, send_from_directory, url_for

from main import app, collection
from main.commons.decorators import parse_args_with
from main.engines.inference import perform_reference, update_inference_to_db
from main.engines.upload import get_metadata_from_csv, handle_jpg_files, handle_zip_file
from main.enums import UPLOAD_FOLDER, Label
from main.libs.utils import get_image_paths
from main.schemas.browse import GetFilteredImagesSchema, UpdateApproval

metadata_df = None
class_names = {0: "challenging-crystal", 1: "crystal", 2: "non-crystal"}


@app.route("/upload", methods=["POST"])
def upload():
    global image_paths
    image_paths = []

    # Check if the 'images' field is present in the request files
    if "images" not in request.files:
        return "No images uploaded", 400
    print(request.files)

    images = request.files.getlist("images")  # Get a list of all uploaded image files

    # Check if any files were selected
    if not images:
        return "No images selected", 400

    global metadata_df
    metadata_df = None

    # Check if the 'metadata' field is present in the request files
    if "metadata" in request.files:
        metadata_file = request.files["metadata"]
        if metadata_file.filename.endswith(".csv"):
            metadata_df = get_metadata_from_csv(metadata_file)

    for file in images:
        # Check if the file has a filename
        if file.filename == "":
            return "One or more files have no filename", 400

        file_path = UPLOAD_FOLDER + "/" + file.filename
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Check if the file is a zip file
        if file.filename.endswith(".zip"):
            # Save the zip file to the upload folder
            handle_zip_file(file, file_path)

        elif file.filename.endswith(".jpg"):
            # Name clashing for jpg uploads
            handle_jpg_files(file, file_path)

    # Redirect to the first result page
    return redirect(url_for("process_images"))


@app.route("/process_images")
def process_images():
    image_paths = get_image_paths()
    predicted_classes_list, probabilities_list = perform_reference(image_paths)
    update_inference_to_db(
        image_paths, predicted_classes_list, probabilities_list, metadata_df
    )
    print(f"predicted_classes_list {predicted_classes_list}")
    print(f"probabilities_list: {probabilities_list}")
    # Redirect to the 'result' page
    return redirect(url_for("result_by_index", index=0))


@app.route("/uploadresult/<int:index>")
def result_by_index(index):
    image_paths = get_image_paths()
    # TODO: Change result page from using image_paths to using database ID's.
    page_index = index % len(image_paths)

    if page_index < 0 or page_index >= len(image_paths):
        return "Invalid index", 400

    file_path = image_paths[page_index]
    file_name = os.path.basename(file_path)

    result_document = collection.find_one({"image_name": file_name})

    # Check if the result document exists
    if result_document is None:
        return f"Result not found for {file_name}", 404

    # Extract the result information from the document
    predicted_class = result_document["assigned_label"]
    probabilities = result_document["probabilities"]
    probabilities_list = list(probabilities.values())

    print(predicted_class)

    print(
        f"Image {file_path}: Predicted class: {predicted_class} with probabilities: {probabilities}."
    )

    # Create the bar chart data
    labels = [Label.CRYSTAL, Label.CHALLENGING_CRYSTAL, Label.NON_CRYSTAL]
    probabilities_list[0], probabilities_list[1] = (
        probabilities_list[1],
        probabilities_list[0],
    )
    colors = [
        "rgb(52, 129, 237)",
        "rgb(130, 232, 133)",
        "rgb(224, 93, 70)",
    ]  # Customize the colors if needed

    # Drawing the stacked bar chart horizontally
    names_col = ["Class", "#", "Probability"]
    plotting_data = [[labels[i], 0, probabilities_list[i]] for i in range(len(labels))]
    plot_df = pd.DataFrame(data=plotting_data, columns=names_col)

    fig = px.bar(
        plot_df,
        x="Probability",
        y="#",
        color="Class",
        title="Classification probabilities",
        orientation="h",
        height=100,
        hover_data={"Class": True, "Probability": True, "#": False},
        color_discrete_sequence=colors,
    )

    fig.update_layout(
        template="simple_white",
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis_range=[0, 1],
        showlegend=False,
    )

    # Set the y axis visibility OFF
    fig.update_yaxes(title="y", visible=False, showticklabels=False)

    # Convert the Figure object to an HTML string
    chart_html = fig.to_html(full_html=False)

    result = {
        "image_path": file_path,
        "image_name": os.path.basename(file_path),
        "db_id": str(result_document["_id"]),
        "predicted_class": predicted_class,
        "approved": result_document["approved"],
        "probabilities": probabilities,
        "chart_html": chart_html,
        "uploaded_batch_index": page_index,
        "uploaded_batch_size": len(image_paths),
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

    return render_template("result_upload.html", result=result)


@app.route("/update_approval/<string:db_id>", methods=["POST"])
def update_approval(db_id):
    if request.method == "POST":
        approved = request.form.get("approve", False)
        label = request.form.get("label", None)

        update_approval = UpdateApproval(approved=approved, label=label)

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
    # Query the MongoDB database to retrieve all entries
    entries = collection.find(args.dict(exclude_none=True))

    # Create a list to store the image data
    images = []

    # Iterate over the entries and extract the image data
    for entry in entries:
        image_data = {
            "db_id": str(entry["_id"]),
            "image_path": entry.get("image_path", ""),
            "image_name": entry.get("image_name", ""),
            "approved": entry.get("approved", ""),
            "assigned_label": entry.get("assigned_label", ""),
        }
        images.append(image_data)

    # Render the 'database.html' template with the image data
    return render_template("database.html", images=images)


@app.route("/browse/database/<string:db_id>")
def result_from_db(db_id):
    result_document = collection.find_one({"_id": ObjectId(db_id)})

    # Check if the result document exists
    if result_document is None:
        return f"Result not found for db_id: {db_id}", 404

    file_path = result_document["image_path"]
    # Extract the result information from the document
    predicted_class = result_document["assigned_label"]
    probabilities = result_document["probabilities"]
    probabilities_list = list(probabilities.values())

    print(predicted_class)

    print(
        f"Image {file_path}: Predicted class: {predicted_class} with probabilities: {probabilities}."
    )

    # Create the bar chart data
    labels = [Label.CRYSTAL, Label.CHALLENGING_CRYSTAL, Label.NON_CRYSTAL]
    labels[0], labels[1] = labels[1], labels[0]
    probabilities_list[0], probabilities_list[1] = (
        probabilities_list[1],
        probabilities_list[0],
    )
    colors = [
        "rgb(52, 129, 237)",
        "rgb(130, 232, 133)",
        "rgb(224, 93, 70)",
    ]  # Customize the colors if needed

    # Drawing the stacked bar chart horizontally
    names_col = ["Class", "#", "Probability"]
    plotting_data = [[labels[i], 0, probabilities_list[i]] for i in range(len(labels))]
    plot_df = pd.DataFrame(data=plotting_data, columns=names_col)

    fig = px.bar(
        plot_df,
        x="Probability",
        y="#",
        color="Class",
        title="Classification probabilities",
        orientation="h",
        height=100,
        hover_data={"Class": True, "Probability": True, "#": False},
        color_discrete_sequence=colors,
    )

    fig.update_layout(
        template="simple_white",
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis_range=[0, 1],
        showlegend=False,
    )

    # Set the y axis visibility OFF
    fig.update_yaxes(title="y", visible=False, showticklabels=False)

    # Convert the Figure object to an HTML string
    chart_html = fig.to_html(full_html=False)

    result = {
        "image_path": file_path,
        "image_name": os.path.basename(file_path),
        "db_id": str(result_document["_id"]),
        "predicted_class": predicted_class,
        "approved": result_document["approved"],
        "probabilities": probabilities,
        "chart_html": chart_html,
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

    return render_template("result_db.html", result=result)
