from huggingface_hub import HfApi

# Initialize the API
api = HfApi()

# Model identifier
model_name = "cross-encoder/stsb-distilroberta-base"

# List files in the model repository
files = api.list_repo_files(repo_id=model_name)

# Print the list of files
for file in files:
    print(file)
