import pickle
import numpy as np

def verify_pickle_file(pkl_path):
    """
    Verify that the pickle file matches LightTBNet format.
   
    Expected format:
    {
        <filename1>: <Numpy Array 256x256>,
        <filename2>: <Numpy Array 256x256>,
        ...
    }
    """
   
    print(f"🔍 Verifying pickle file: {pkl_path}\n")
   
    try:
        # Load pickle file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
       
        print("✅ Pickle file loaded successfully")
       
        # Check if it's a dictionary
        if not isinstance(data, dict):
            print(f"❌ ERROR: Expected dict, but got {type(data)}")
            return False
       
        print(f"✅ Data is a dictionary")
        print(f"✅ Total images: {len(data)}")
       
        # Check each image
        all_correct = True
        errors = []
       
        for idx, (filename, img) in enumerate(data.items()):
            # Check if value is numpy array
            if not isinstance(img, np.ndarray):
                errors.append(f"   ❌ {filename}: Not a numpy array (got {type(img)})")
                all_correct = False
                continue
           
            # Check image shape (should be 256x256)
            if img.shape != (256, 256):
                errors.append(f"   ❌ {filename}: Wrong shape {img.shape}, expected (256, 256)")
                all_correct = False
                continue
           
            # Print first 5 as examples
            if idx < 5:
                print(f"   ✅ {filename}: shape {img.shape}, dtype {img.dtype}")
       
        # Show any errors
        if errors:
            print(f"\n⚠️  Found {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(error)
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
       
        # Final summary
        print(f"\n📊 Summary:")
        print(f"   Total images: {len(data)}")
        print(f"   All images correct: {'✅ YES' if all_correct else '❌ NO'}")
       
        if all_correct:
            print(f"\n✨ Pickle file is in CORRECT format!")
            return True
        else:
            print(f"\n❌ Pickle file has ERRORS!")
            return False
           
    except Exception as e:
        print(f"❌ ERROR loading pickle file: {e}")
        return False


if __name__ == "__main__":
    pkl_path = "./data/images.pkl"
    verify_pickle_file(pkl_path) 