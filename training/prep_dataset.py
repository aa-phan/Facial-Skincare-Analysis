import pandas as pd
import os
import requests
import time

headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/58.0.3029.110 Safari/537.3"),
    # Optionally include Accept header
    "Accept": "image/webp,image/*,*/*;q=0.8"
}

df = pd.read_csv('fitzpatrick17k.csv')
df = df[['label', 'url']]
df.to_csv('labels.csv', index=False)
os.makedirs('data/fitzpatrick17k_data', exist_ok=True)

# Loop through each row in the dataframe
for idx, row in df.iterrows():
    time.sleep(0.2)

    label = row.label
    url = row.url

    image_path = os.path.join("data/fitzpatrick17k_data", f"{idx}.jpg")
    try:
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=2)
        print(f"Downloaded {idx}: [{time.time() - start_time:.2f}]s")
    except requests.RequestException as e:
        print(f"Failed to download image from {url} due to {e}")
        continue

    with open(image_path, "wb") as img_file:
        img_file.write(response.content)

print("Done!")
