from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

# 1. Load a pretrained CrossEncoder model
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# We want to compute the similarity between the query sentence...
# query = "honda 2023 fuel injectors"
query = "hands fuel injectars"
# ... and all sentences in the corpus
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

# 2. We rank all sentences in the corpus for the query
ranks = model.predict([[query, sentence] for sentence in corpus])

# Sort the scores in decreasing order to get the corpus indices
ranked_indices = np.argsort(ranks)[::-1]

# Print the ranked results
print("Query: ", query)
print("Ranked Results:")
for idx in ranked_indices:
    print(f"{ranks[idx]:.2f}\t{corpus[idx]}")
