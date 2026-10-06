import os
import pandas as pd
from pymongo import MongoClient
from pymongo import UpdateOne
from dotenv import load_dotenv

load_dotenv()

# Read cleaned data
df = pd.read_json("SmartMovie/data/raw.json")

# Connect to MongoDB
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)

db = client["smartmovie"]
collection = db["movies"]

# Prepare operations
operations = []

for movie in df.to_dict("records"):
    operations.append(
        UpdateOne(
            {"movie_id": movie["movie_id"]},
            {"$set": movie},
            upsert=True
        )
    )

# Load into MongoDB
collection.bulk_write(operations)

print(f"{len(operations)} movies loaded into MongoDB")