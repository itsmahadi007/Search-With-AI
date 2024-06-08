from huggingface_hub import hf_hub_download

# Model identifier from the Hugging Face model hub
model_name = "cross-encoder/stsb-distilroberta-base"

# List of files to download (based on the available files)
files_to_download = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt"
]

# Download each file to the current directory
for file in files_to_download:
    try:
        hf_hub_download(
            repo_id=model_name,
            filename=file,
            local_dir=".",  # Save to current directory
            force_download=True  # Ensure fresh download
        )
        print(f"Downloaded {file}")
    except Exception as e:
        print(f"Failed to download {file}: {e}")
