import os
import time
import requests
import asyncio
import aiohttp
import pandas as pd

# Keep using requests for the CSV
print("Downloading Fitzpatrick17k dataset...")
csv_url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/refs/heads/main/fitzpatrick17k.csv"
headers = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/58.0.3029.110 Safari/537.3"),
    "Accept": "image/webp,image/*,*/*;q=0.8"
}

# Download CSV
response = requests.get(csv_url)
with open("fitzpatrick17k.csv", "wb") as file:
    file.write(response.content)

# Prepare DataFrame
df = pd.read_csv("fitzpatrick17k.csv")
df = df[["label", "url"]].dropna()  # Drop rows missing label or url
os.makedirs("data/fitzpatrick17k_data", exist_ok=True)


async def download_image(session, idx, label, url, headers):
    """
    Downloads a single image asynchronously using aiohttp.
    Returns the row data if successful, or None if failed.
    """
    start_time = time.time()
    image_path = os.path.join("data/fitzpatrick17k_data", f"{idx}.jpg")
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            response.raise_for_status()
            content = await response.read()

        # Save to disk
        with open(image_path, "wb") as img_file:
            img_file.write(content)

        print(f"Downloaded {idx} in [{time.time() - start_time:.2f}]s")
        return {"label": label, "url": url}

    except Exception as e:
        print(f"Failed to download image {idx} from {url} due to {e}")
        with open("failed_downloads.log", "a") as log_file:
            log_file.write(f"Failed to download image {idx} from {url} due to {e}\n")
        return None


async def download_all_images(df, headers):
    """
    Creates and awaits tasks for all images in the DataFrame.
    Returns a list of rows for successful downloads.
    """
    tasks = []

    # Limit connections if needed; adjust `limit` for concurrency preference
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, row in df.iterrows():
            # Create a coroutine for each image
            tasks.append(
                asyncio.create_task(download_image(session, idx, row.label, row.url, headers))
            )
            await asyncio.sleep(0.01)

        # Gather results (return_exceptions=True to collect any errors but not stop all)
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None and any exceptions
    successful = []
    for res in results:
        if isinstance(res, dict):
            successful.append(res)
    return successful


def main():
    # Run the async download logic
    loop = asyncio.get_event_loop()
    successful_rows = loop.run_until_complete(download_all_images(df, headers))

    # Save successful rows
    if successful_rows:
        df_success = pd.DataFrame(successful_rows, columns=["label", "url"])
        df_success.to_csv("labels_fitzpatrick17k.csv", index=False)

    # Clean up the CSV
    os.remove("fitzpatrick17k.csv")

    print("Done!")


if __name__ == "__main__":
    main()
