import faiss
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import transforms as T
import huggingface_hub
from gradio_client import Client
import base64
import io
from gradio import components

import pymongo

from db import get_gallery_embeddings



def get_similiarity_l2(embeddings_gallery, embedding_query, k=10):

    index = faiss.IndexFlatL2(embeddings_gallery.shape[1])
    index.add(embeddings_gallery)

    # Reshape the query_embedding to match the gallery vectors
    embedding_query= np.array(embedding_query).reshape(1, -1)  

    scores, indices = index.search(embedding_query, k) 

    flat_indices = [int(index) for sublist in indices for index in sublist]

    return flat_indices

def serialize_image(image):
    """Serializes a components.Image object to a base64 string."""
    image_bytes = image.tobytes()
    base64_encoded_image = base64.encodebytes(image_bytes)
    return base64_encoded_image.decode()

def get_embedding(img_path):
    # image = components.Image(img_path)

    image = Image.open(img_path)
    image_base64 = base64.b64encode(image.tobytes()).decode('utf-8')

    # serialized_image = serialize_image(image)

    client = Client("https://yasirapunsith-vpr-v3.hf.space/--replicas/p4mqt/")
    embedding = client.predict(
            {"image": image_base64},	# filepath  in 'image' Image component
            api_name="/predict")
    
    return embedding["predictions"]


def get_similar_images_list():
    embedding = get_embedding("abiding-warm-buffalo-of-vitality.jpeg")
    
    # embedding = 

    return run(embedding["predictions"])

def run(embedding_query):
    # def run(embedding_query):
    # Define your MongoDB connection parameters
    mongo_uri = "mongodb://localhost:27017"  # Replace with your MongoDB URI
    database_name = "vpr_db"  # Replace with your database name
    collection_name = "images"  # Replace with your collection name

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    embeddings_gallery = get_gallery_embeddings(collection)


    print("got the indices")


    indices = get_similiarity_l2(embeddings_gallery, embedding_query, k=10)

    images_list = []

    for image_id in indices:
        # Query the MongoDB collection to retrieve the document with the specified ID
        document = collection.find_one({"id": image_id})

        # Check if the document exists
        if document:
            # Get the base64-encoded image data from the document
            image_base64 = document.get("image")
            images_list.append(image_base64)
        else:
            print("Image not found.")

    # Close the MongoDB connection
    client.close()
    return images_list  

