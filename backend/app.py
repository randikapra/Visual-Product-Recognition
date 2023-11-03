from flask import Flask, jsonify, request
from flask_wtf import FlaskForm 
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_cors import CORS

import os

import pymongo
from pymongo import MongoClient

import numpy as np
import faiss


from utilities import get_embedding, get_similar_images_list, run

app = Flask(__name__)

# CORS(app)

CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5173"}})
# Initialize CORS with custom settings allowing requests from both Streamlit URLs
CORS(app, resources={
    r"/predict": {
        "origins": "*"
    }
})


mongo_uri = "mongodb://localhost:27017"  # Replace with your MongoDB URI
database_name = "vpr_db"  # Replace with your database name
collection_name = "images"  # Replace with your collection name

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]


# Define a function to get an image from MongoDB by its name
def get_image_by_index(img_id):
    return collection.find_one({"id": img_id})


app.config["SECRET_KEY"] = "supersecretkey" 
app.config["UPLOADED_PHOTOS_DEST"] = "static/files"

photos = UploadSet("photos", IMAGES)
# go through all the UploadSets and get their confighuration and 
# store the configuration on the app
configure_uploads(app, photos) 


class UploadFileField(FlaskForm):
    file = FileField(
        validators=[
            FileAllowed(photos, "only images are allowed"),
            FileRequired("File field should not be empty")
    ])
    submit = SubmitField("Upload File")


@app.route("/api/get_similar_images", methods=["POST"])
def get_similar_images():
    # Get the uploaded image file
    uploaded_file = request.files["file"]

    # Check if the file exists and has an allowed file extension
    if uploaded_file and allowed_file(uploaded_file.filename):
        # Save the uploaded file to a specific location
        file_path = os.path.join(app.config["UPLOADED_PHOTOS_DEST"], secure_filename(uploaded_file.filename))
        uploaded_file.save(file_path)

        # Retrieve the actual images from MongoDB and encode them as base64
        similar_images_data = get_similar_images_list()
        print("inside app")
        # print(similar_images_data)
        count = 1
        similarImages = []
        for image in similar_images_data:
            if image:
                # Convert image data to base64
                # image_base64 = image_data['data'].decode('utf-8')
                similarImages.append({
                    'image_name': str(count),
                    'image_base64': image
                })
                count += 1

        similar_images_data = {'similarImages': similarImages}
        return jsonify(similar_images_data)
    else:
        return jsonify({"error": "Invalid file"})

    
def allowed_file(filename):
    # Check if the file extension is allowed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ["jpg", "jpeg", "png", "gif"]
    

if __name__ == "__main__":
    app.run(port=5000)


