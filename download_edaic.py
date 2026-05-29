import os
import shutil
import re
import urllib.request
import tarfile
from urllib.error import URLError

base_url = "https://dcapswoz.ict.usc.edu/wwwedaic/"
data_url = base_url + "data/"
output_dir = "wwwedaic"
data_output_dir = os.path.join(output_dir, "data")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

def download_file(url, out_path):
    print(f"Downloading {url} to {out_path}...")
    try:
        urllib.request.urlretrieve(url, out_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# Download root files
print("Downloading root files...")
for file in ["metadata_mapped.csv", "labels2019.tar.gz", "E-DAIC%20Manual.pdf"]:
    download_file(base_url + file, os.path.join(output_dir, file.replace("%20", " ")))

# Extract labels
labels_tar = os.path.join(output_dir, "labels2019.tar.gz")
if os.path.exists(labels_tar):
    with tarfile.open(labels_tar, "r:gz") as tar:
        tar.extractall(path=output_dir)

# Fetch data directory HTML
print("Fetching data directory listing...")
try:
    req = urllib.request.Request(data_url)
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
    
    # Extract all .tar.gz links
    links = re.findall(r'href="(\d+_P\.tar\.gz)"', html)
    print(f"Found {len(links)} participant archives.")
    
    for link in links:
        out_path = os.path.join(data_output_dir, link)
        participant_dir = os.path.join(data_output_dir, link.replace('.tar.gz', ''))
        
        if not os.path.exists(participant_dir):
            if os.path.exists(out_path):
                # Delete potentially partial download from previous run
                os.remove(out_path)
            
            success = download_file(data_url + link, out_path)
            if success:
                print(f"Extracting {link}...")
                try:
                    with tarfile.open(out_path, "r:gz") as tar:
                        tar.extractall(path=data_output_dir)
                    print(f"Extracted. Deleting {link} to save space...")
                    os.remove(out_path)
                    features_dir = os.path.join(participant_dir, "features")
                    if os.path.exists(features_dir):
                        shutil.rmtree(features_dir, ignore_errors=True)
                        print(f"Deleted bloated features directory for {link}")
                except Exception as e:
                    print(f"Extraction failed for {link}: {e}")
        else:
            print(f"Skipping {link} (already extracted)")
            
    print("Download and extraction complete.")
except URLError as e:
    print(f"Failed to fetch directory listing: {e}")

