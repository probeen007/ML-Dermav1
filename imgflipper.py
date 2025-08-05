import os
from PIL import Image

def flip_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                img = Image.open(file_path)
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                output_path = os.path.join(folder_path, f"flipped_{filename}")
                flipped_img.save(output_path)
                print(f"Flipped and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    folder_path = input("Enter the folder path containing images: ")
    flip_images_in_folder(folder_path)

if __name__ == "__main__":
    main()
