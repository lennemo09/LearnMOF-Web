from flask import jsonify, render_template

from main import app


@app.route("/pings", methods=["GET"])
def ping():
    return jsonify({})


@app.route("/ready", methods=["GET"])
def is_ready():
    return jsonify({})


@app.route("/")
def index():
    return render_template("index.html")
