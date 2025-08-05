import os
import requests
import time
import threading
from bs4 import BeautifulSoup

# Function to fetch image URLs from a webpage
def get_image_urls(url, class_filter):
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch page: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.find_all("img", class_=class_filter)  # Find images with the given class
    
    image_urls = []
    for img in images:
        src = img.get("src", "")
        
        if src:
            full_url = src if src.startswith("http") else requests.compat.urljoin(url, src)  
            image_urls.append(full_url)
    
    return image_urls

# Function to download an image using streaming
def download_image(url, save_path, index, retries=3):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        
        for attempt in range(retries):
            print(f"Attempting to download {url} (Attempt {attempt+1}/{retries})")
            response = requests.get(url, headers=headers, stream=True)
            
            if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                file_path = os.path.join(save_path, f"image_{index+1}.jpg")
                
                with open(file_path, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        if chunk:
                            img_file.write(chunk)
                
                print(f"Downloaded: {file_path}")
                break  
            else:
                print(f"Not a valid image or error downloading: {url}")
            
            time.sleep(5)
    
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Function to download all images using threading
def download_images(image_urls, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    threads = []
    for i, url in enumerate(image_urls):
        thread = threading.Thread(target=download_image, args=(url, save_path, i))
        threads.append(thread)
        thread.start()
        time.sleep(1) 
    
    for thread in threads:
        thread.join()

# Main function
def main():
    url = "https://www.istockphoto.com/search/more-like-this/1493956084"  # Target webpage
    save_path = "D:/Professional/Derma/SkinDisease/train/Acne"

    class_filter = "eRezkZyQZevxFgt1jMkt"  # Class to filter images
    
    save_path = save_path.replace("\\", "/")
    
    print("Fetching image links...")
    image_urls = get_image_urls(url, class_filter)
    
    if not image_urls:
        print("No images found.")
    else:
        print(f"Found {len(image_urls)} images. Downloading...")
        download_images(image_urls, save_path)
        print("Download complete.")

if __name__ == "__main__":
    main()
