import os
import cv2
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def convert_dataset_to_lighttbnet(
    image_folder,
    output_dir="./data",
    csv_name="dataset.csv",
    pkl_name="images.pkl",
    image_size=64 #changed from 256
):
    """
    Convert image dataset to LightTBNet format.
   
    Parameters:
    -----------
    image_folder : str
        Path to folder containing your images
    output_dir : str
        Directory to save CSV and pickle files
    csv_name : str
        Name of the CSV file to create
    pkl_name : str
        Name of the pickle file to create
    image_size : int
        Size to resize images to (default 256x256)
   
    Instructions:
    -----------
    1. Organize your images in folders by class:
       image_folder/
       ├── TB/
       │   ├── patient_001_age_25_M_image1.jpg
       │   ├── patient_001_age_25_M_image2.jpg
       │   └── ...
       └── Normal/
           ├── patient_002_age_30_F_image1.jpg
           └── ...
   
    2. Filename format: patient_ID_age_XX_GENDER_description.jpg
       (The script will parse patient_id, age, and sex from filename)
   
    3. Run this script and it will create:
       - data/dataset.csv (with columns: patient_id, age_yo, sex, filename, TB_class, split, fold_cv)
       - data/images.pkl (numpy arrays 256x256)
    """
   
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
   
    # Data containers
    data_records = []
    images_dict = {}
   
    # Class mapping
    class_folders = {"TB Chest X-rays": 1, "Normal Chest X-rays": 0}  # Adjust if your folders have different names
   
    print(f"🔍 Scanning images from: {image_folder}")
   
    # Iterate through class folders
    for class_name, class_label in class_folders.items():
        class_path = os.path.join(image_folder, class_name)
       
        if not os.path.exists(class_path):
            print(f"⚠️  Warning: Folder not found: {class_path}")
            continue
       
        print(f"\n📁 Processing class: {class_name}")
       
        # Iterate through images in class folder
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
       
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
           
            try:
                # Read and resize image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"  ❌ Failed to read: {img_file}")
                    continue
               
                img = cv2.resize(img, (image_size, image_size))
               
                # Parse filename to extract patient info
                # Expected format: patient_ID_age_XX_GENDER_description.jpg
                filename_parts = img_file.split('_')
               
                try:
                    patient_id = filename_parts[1]  # After 'patient'
                    age = int(filename_parts[3])  # After 'age'
                    sex = filename_parts[4]  # Gender (M/F)
                except (IndexError, ValueError):
                    print(f"  ⚠️  Could not parse filename: {img_file}")
                    print(f"     Expected format: patient_ID_age_XX_GENDER_description.jpg")
                    patient_id = f"unknown_{idx}"
                    age = 0
                    sex = "U"
               
                # Create record
                record = {
                    'patient_id': patient_id,
                    'age_yo': age,
                    'sex': sex,
                    'filename': img_file,
                    'TB_class': class_label,
                    'split': 'train',  # You can modify this logic (train/val/test)
                    'fold_cv': 0  # You can modify this for cross-validation
                }
               
                data_records.append(record)
                images_dict[img_file] = img
               
                print(f"  ✓ {img_file} - Patient {patient_id}, Age {age}, {sex}")
               
            except Exception as e:
                print(f"  ❌ Error processing {img_file}: {str(e)}")
                continue
   
    # Create DataFrame
    print(f"\n💾 Creating CSV with {len(data_records)} records...")
    df = pd.DataFrame(data_records)
   
    # Ensure correct column order
    column_order = ['patient_id', 'age_yo', 'sex', 'filename', 'TB_class', 'split', 'fold_cv']
    df = df[column_order]
   
    # Save CSV
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")
    print(f"\nFirst few rows:")
    print(df.head())
   
    # Save pickle file with images
    print(f"\n💾 Creating pickle file with {len(images_dict)} images...")
    pkl_path = os.path.join(output_dir, pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(images_dict, f)
    print(f"✅ Pickle file saved: {pkl_path}")
   
    print(f"\n✨ Dataset conversion complete!")
    print(f"   Total images: {len(images_dict)}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Output files ready in: {output_dir}/")


if __name__ == "__main__":
    # MODIFY THESE PATHS FOR YOUR DATASET
    image_folder = "/home/raniya/Downloads/Dataset of TB CXR img" # Change this to your image folder path
    output_dir = "./data"  # Where to save CSV and pickle
   
    convert_dataset_to_lighttbnet(
        image_folder=image_folder,
        output_dir=output_dir,
        csv_name="dataset.csv",
        pkl_name="images.pkl",
        image_size=64 #changed
    )