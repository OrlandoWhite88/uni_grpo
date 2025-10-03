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
