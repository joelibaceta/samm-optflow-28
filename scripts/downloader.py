# downloader.py
"""
Script to download the SAMM v2 dataset from Kaggle and extract it into a local folder.
Requires the Kaggle API client and a configured ~/.kaggle/kaggle.json.
Usage:
    python downloader.py \
      --dataset muhammadzamancuiisb/samm-v2 \
      --output datasets/SAMM2 \
      [--unzip]
"""
import os
import argparse
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(dataset: str, output_dir: str, unzip: bool = False):
    """
    Download a Kaggle dataset and optionally unzip.

    Args:
        dataset (str): Kaggle dataset identifier (e.g. 'muhammadzamancuiisb/samm-v2').
        output_dir (str): Directory to download files into.
        unzip (bool): If True, extract all ZIP files and remove them.
    """
    api = KaggleApi()
    api.authenticate()

    os.makedirs(output_dir, exist_ok=True)
    print(f"🔄 Downloading dataset '{dataset}' to '{output_dir}'...")
    api.dataset_download_files(dataset, path=output_dir, quiet=False)

    if unzip:
        print(f"🔄 Extracting ZIP files in '{output_dir}'...")
        for fname in os.listdir(output_dir):
            if fname.endswith('.zip'):
                zip_path = os.path.join(output_dir, fname)
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(output_dir)
                    os.remove(zip_path)
                    print(f"✅ Extracted and removed '{fname}'")
                except zipfile.BadZipFile:
                    print(f"⚠️ Skipping invalid zip: {fname}")
    print("✅ Download complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download a Kaggle dataset and extract it.')
    parser.add_argument('-d', '--dataset', required=True,
                        help="Kaggle dataset identifier (e.g. 'username/dataset-name')")
    parser.add_argument('-o', '--output', default='datasets/SAMM2',
                        help='Local directory to download into')
    parser.add_argument('-u', '--unzip', action='store_true',
                        help='Extract ZIP files and remove archives')
    args = parser.parse_args()

    download_dataset(args.dataset, args.output, args.unzip)