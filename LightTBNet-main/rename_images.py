import os
from pathlib import Path

def rename_images_batch(image_folder):
    """
    Automatically rename all images to LightTBNet format.
   
    Format: patient_ID_age_XX_GENDER_originalname.jpg
   
    Parameters:
    -----------
    image_folder : str
        Path to your main image folder containing "Normal chest x-rays" and "TB chest x-rays"
    """
   
    folder_mapping = {
        "Normal Chest X-rays": ("Normal", 0),
        "TB Chest X-rays": ("TB", 1)
    }
   
    total_renamed = 0
   
    for folder_name, (class_type, class_id) in folder_mapping.items():
        folder_path = os.path.join(image_folder, folder_name)
       
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue
       
        print(f"\n📁 Processing: {folder_name}")
       
        # Get all image files
        image_files = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
       
        print(f"   Found {len(image_files)} images to rename")
       
        # Counter for patient IDs
        patient_counter = 1
       
        for idx, old_filename in enumerate(image_files, 1):
            # Create new filename
            # Format: patient_XXX_age_0_U_originalname.jpg
            # (U = Unknown, since we don't have actual patient data)
           
            file_ext = os.path.splitext(old_filename)[1]
            new_filename = f"patient_{patient_counter:03d}_age_0_U_{old_filename}"
           
            old_path = os.path.join(folder_path, old_filename)
            new_path = os.path.join(folder_path, new_filename)
           
            try:
                os.rename(old_path, new_path)
                if idx % 100 == 0:  # Print every 100 files
                    print(f"   ✓ Renamed {idx}/{len(image_files)}")
                total_renamed += 1
                patient_counter += 1
            except Exception as e:
                print(f"   ❌ Error renaming {old_filename}: {e}")
   
    print(f"\n✨ Done! Renamed {total_renamed} images total")
    print(f"\n📝 Examples of new filenames:")
    print(f"   patient_001_age_0_U_originalimage.jpg")
    print(f"   patient_002_age_0_U_xray.jpg")
    print(f"\n💡 Note: age_0 and U (Unknown) are placeholders.")
    print(f"   You can manually edit the CSV later if you have actual patient data.")


if __name__ == "__main__":
    # CHANGE THIS TO YOUR IMAGE FOLDER PATH
    image_folder = "/home/raniya/Downloads/Dataset of TB CXR img"
   
    print(f"🔄 Starting batch rename...")
    print(f"📂 Image folder: {image_folder}\n")
   
    rename_images_batch(image_folder)