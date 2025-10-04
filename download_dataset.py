import gdown

# Google Drive file ID for dataset
file_id = "112cHMg-FqR2m_HNIZMsgn0RbjhX1EtKP"

# Output file name
output = "dataset.jsonl"

# Download the file
print(f"Downloading dataset to {output}...")
gdown.download(id=file_id, output=output, quiet=False, use_cookies=False)
print(f"Download complete! File saved as {output}")

# Google Drive file ID for LoRA adapter
lora_file_id = "1SXRDMKrlY3HBeHn83bUlDjbZ_c9MlT"

# Output file name for LoRA
lora_output = "lora_adapter.zip"  # Assuming zip, adjust if needed

# Download the LoRA file
print(f"Downloading LoRA adapter to {lora_output}...")
try:
    gdown.download(id=lora_file_id, output=lora_output, quiet=False, use_cookies=False)
    print(f"Download complete! LoRA file saved as {lora_output}")
except Exception as e:
    print(f"LoRA download failed: {e}")
    print("Please ensure the file is set to 'Anyone with the link' on Google Drive.")

# Google Drive folder ID for model
folder_id = "1QKsmdbFOZGtwyQm4w4EuGWYdgwpyF1RV"

# Output folder name
output_folder = "grpo-adapter-step-20"

# Download the folder
print(f"Downloading model folder to {output_folder}...")
try:
    gdown.download_folder(id=folder_id, output=output_folder, quiet=False, use_cookies=False)
    print(f"Download complete! Folder saved as {output_folder}")
except Exception as e:
    print(f"Model folder download failed: {e}")
    print("Please ensure the folder is set to 'Anyone with the link' on Google Drive.")
