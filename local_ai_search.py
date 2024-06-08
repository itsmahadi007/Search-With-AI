from sentence_transformers import CrossEncoder
import numpy as np
import os

# Define the local directory where the model files are stored
model_path = os.getcwd()

# Load the pretrained model from the local directory
model = CrossEncoder(model_path)

# Define the query and corpus
query = "hands fuel injectars"
corpus = [
    "Fuel injectors for Honda Accord 2023",
    "Carburetor for 2021 Toyota Camry",
    "Engine for Ford F-150 2020",
    "Brake pads for Honda Civic 2018",
    "Fuel pump for Toyota Corolla 2019",
    "Transmission for BMW X5 2022",
    "Fuel injectors for Honda Civic 2020",
    "Alternator for Nissan Altima 2021",
    "Battery for Chevrolet Impala 2019",
]

# Rank the sentences
ranks = model.predict([[query, sentence] for sentence in corpus])
ranked_indices = np.argsort(ranks)[::-1]

# Print the ranked results
print("Query: ", query)
print("Ranked Results:")
for idx in ranked_indices:
    print(f"{ranks[idx]:.2f}\t{corpus[idx]}")
