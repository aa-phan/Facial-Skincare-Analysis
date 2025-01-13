import pandas as pd
import os
import requests
import time

headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/58.0.3029.110 Safari/537.3"),
    "Accept": "image/webp,image/*,*/*;q=0.8"
}

df = pd.read_csv("fitzpatrick17k.csv")
df = df[["label", "url"]]

os.makedirs("data/fitzpatrick17k_data", exist_ok=True)

successful_rows = []
for idx, row in df.iterrows():
    time.sleep(0.2)

    label = row.label
    url = row.url

    image_path = os.path.join("data/fitzpatrick17k_data", f"{idx}.jpg")

    try:
        start_time = time.time()
        response = requests.get(url, headers=headers, timeout=2)
        response.raise_for_status()  # Raise an exception for bad status codes
        print(f"Downloaded {idx} in [{time.time() - start_time:.2f}]s")

        with open(image_path, "wb") as img_file:
            img_file.write(response.content)

        successful_rows.append(row)

    except requests.RequestException as e:
        print(f"Failed to download image from {url} due to {e}")
        with open("failed_downloads.log", "a") as log_file:
            log_file.write(f"Failed to download image from {url} due to {e}\n")


df_success = pd.DataFrame(successful_rows, columns=["label", "url"])
df_success.to_csv("labels_fitzpatrick17k.csv", index=False)

print("Done!")
