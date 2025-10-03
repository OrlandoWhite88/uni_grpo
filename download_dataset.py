import gdown

# Google Drive file ID
file_id = "112cHMg-FqR2m_HNIZMsgn0RbjhX1EtKP"

# Construct the download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Output file name
output = "dataset.jsonl"

# Download the file
print(f"Downloading dataset to {output}...")
gdown.download(url, output, quiet=False)
print(f"Download complete! File saved as {output}")

# Google Drive folder ID for model
folder_id = "1QKsmdbFOZGtwyQm4w4EuGWYdgwpyF1RV"

# Output folder name
output_folder = "model"

# Download the folder
print(f"Downloading model folder to {output_folder}...")
gdown.download_folder(folder_id, output_folder, quiet=False)
print(f"Download complete! Folder saved as {output_folder}")
