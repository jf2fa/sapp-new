import os
import pandas as pd
import torch
import pickle
from sentence_transformers import SentenceTransformer

# Load the model
def load_model():
    model = SentenceTransformer('all-mpnet-base-v2')
    return model

model = load_model()

# Function to encode rows using the model
def encode_rows(rows):
    return model.encode(rows, convert_to_tensor=True)

# Path to the folder containing the CSV files
folder_path = 'path_to_your_csv_folder'

# Lists to hold embeddings and metadata
embeddings = []
metadata = []

# Process each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        rows = [row.to_json() for _, row in df.iterrows()]
        
        # Encode rows and store embeddings and metadata
        encoded_rows = encode_rows(rows)
        embeddings.extend(encoded_rows)
        metadata.extend([{'file': filename, 'row': row} for row in rows])

# Convert embeddings to tensor
embeddings_tensor = torch.stack(embeddings) if embeddings else torch.empty(0, dtype=torch.float32)

# Save the embeddings and metadata to .pkl files
with open('precomputed_embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_tensor.tolist(), f)

with open('precomputed_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Precomputed embeddings and metadata have been saved.")
