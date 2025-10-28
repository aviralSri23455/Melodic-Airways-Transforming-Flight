#!/usr/bin/env python3
"""
Download OpenFlights data and save locally for import
"""

import requests
import os
from app.core.config import settings

def download_openflights_data():
    """Download OpenFlights data files"""
    base_url = settings.OPENFLIGHTS_BASE_URL

    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    files_to_download = [
        ("airports.dat", "airports.dat"),
        ("routes.dat", "routes.dat")
    ]

    print("📥 Downloading OpenFlights data...")

    for filename, url_part in files_to_download:
        url = f"{base_url}/{url_part}"
        file_path = os.path.join(data_dir, filename)

        print(f"Downloading {filename}...")
        response = requests.get(url)

        if response.status_code == 200:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"✅ Saved {filename} to {file_path}")
        else:
            print(f"❌ Failed to download {filename}")

    print("🎉 Download complete!")
    print(f"Data files saved in: {os.path.abspath(data_dir)}")
    print("")
    print("To import the data, run:")
    print("python scripts/etl_openflights.py data/airports.dat data/routes.dat")

if __name__ == "__main__":
    download_openflights_data()
