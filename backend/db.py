import pymongo
import json
import numpy as np


def get_gallery_embeddings(collection):
    cursor = collection.find({})

    # Initialize a list to store feature vectors
    feature_vectors_list = []

    # Iterate through the documents and extract the feature vectors
    for document in cursor:
        feature_vector = document.get("feature_vector")
        if feature_vector is not None:
            feature_vectors_list.append(np.array(feature_vector))

    return np.array(np.array(feature_vectors_list))



if (__name__ == "__main__"):
    # Define your MongoDB connection parameters
    mongo_uri = "mongodb://localhost:27017"  # Replace with your MongoDB URI
    database_name = "vpr_db"  # Replace with your database name
    collection_name = "images"  # Replace with your collection name

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_uri)
    db = client[database_name]
    collection = db[collection_name]

    # Load the JSON file
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)

    # Insert the JSON data into the collection
    collection.insert_many(data)

    # Close the MongoDB connection
    client.close()

