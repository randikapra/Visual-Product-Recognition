import pymongo
import base64
from PIL import Image
import io

from utilities import get_embedding, get_similiarity_l2
from db import get_gallery_embeddings

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


    img_path = r"static\files\abiding-warm-buffalo-of-vitality.jpeg"

    # embedding_query = get_embedding(img_path)

    embeddings_gallery = get_gallery_embeddings(collection)


    print("got the indices")


    indices = get_similiarity_l2(embeddings_gallery, embedding_query, k=10)

    # Define the image's unique identifier (ID)
    image_id = 934  # Replace with the actual ID of the image you want to display

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
