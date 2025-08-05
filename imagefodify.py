import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image

def augment_images(folder_path):
    transform = A.Compose([
        A.Rotate(limit=100, p=0.7),
        A.RandomBrightnessContrast(p=0.7),
        A.GaussianBlur(p=0.4),
        A.RandomScale(scale_limit=0.2, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.7),
        A.Affine(shear=10, p=0.5)
    ])

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            try:
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = transform(image=img)['image']
                augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                output_path = os.path.join(folder_path, f"aug_{filename}")
                cv2.imwrite(output_path, augmented)
                print(f"Augmented and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

def main():
    folder_path = "D:/Professional/Derma/SkinDisease/train/Warts"  # Hardcoded path to the images
    augment_images(folder_path)

if __name__ == "__main__":
    main()
