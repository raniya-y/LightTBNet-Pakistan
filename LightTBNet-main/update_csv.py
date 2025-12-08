# updating splits and cross-validation 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def update_csv_with_splits(csv_path, output_path=None, test_size=0.2, n_splits=5):
    """
    Update CSV with train/test split and cross-validation folds.
   
    Parameters:
    -----------
    csv_path : str
        Path to your dataset.csv
    output_path : str
        Where to save the updated CSV (default: overwrites original)
    test_size : float
        Proportion of data for testing (0.2 = 20% test, 80% train)
    n_splits : int
        Number of cross-validation folds (default: 5)
   
    Example output:
    ---------------
    split column values: 'train' or 'test'
    fold_cv column values: 0, 1, 2, 3, 4 (for 5-fold cross-validation)
    """
   
    # Read CSV
    print(f"📖 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total images: {len(df)}")
   
    # First, split into train and test
    print(f"\n🔀 Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
   
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=42,
        stratify=df['TB_class']  # Keeps TB/Normal ratio balanced
    )
   
    # Create split column
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'
   
    print(f"   Train: {len(train_idx)} images")
    print(f"   Test:  {len(test_idx)} images")
   
    # Assign cross-validation folds (only to training data)
    print(f"\n🔄 Assigning {n_splits}-fold cross-validation...")
   
    df['fold_cv'] = -1  # Initialize all as -1
   
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
   
    train_data = df.loc[train_idx]
    fold_counter = 0
   
    for train_fold_idx, val_fold_idx in kfold.split(train_data):
        df.loc[train_idx[val_fold_idx], 'fold_cv'] = fold_counter
        fold_counter += 1
   
    # Test data gets fold_cv = -1 (not used in cross-validation)
    df.loc[test_idx, 'fold_cv'] = -1
   
    print(f"   Folds assigned (0-{n_splits-1})")
   
    # Show summary
    print(f"\n📊 Summary:")
    print(f"   Train with folds:")
    for fold in range(n_splits):
        count = len(df[(df['split'] == 'train') & (df['fold_cv'] == fold)])
        print(f"     Fold {fold}: {count} images")
   
    test_count = len(df[df['split'] == 'test'])
    print(f"   Test: {test_count} images")
   
    # Show class distribution
    print(f"\n📋 Class distribution:")
    print(f"   Train - TB: {len(df[(df['split'] == 'train') & (df['TB_class'] == 1)])} | Normal: {len(df[(df['split'] == 'train') & (df['TB_class'] == 0)])}")
    print(f"   Test  - TB: {len(df[(df['split'] == 'test') & (df['TB_class'] == 1)])} | Normal: {len(df[(df['split'] == 'test') & (df['TB_class'] == 0)])}")
   
    # Save
    if output_path is None:
        output_path = csv_path
   
    df.to_csv(output_path, index=False)
    print(f"\n✅ Updated CSV saved: {output_path}")
   
    # Show sample rows
    print(f"\n📝 Sample rows:")
    print(df[['filename', 'TB_class', 'split', 'fold_cv']].head(10))


if __name__ == "__main__":
    # Path to your CSV
    csv_path = "./data/dataset.csv"
   
    # Optional: save to a different location
    # output_path = "./data/dataset_updated.csv"
   
    # Update with:
    # - 80% train, 20% test
    # - 5-fold cross-validation
    update_csv_with_splits(
        csv_path=csv_path,
        test_size=0.2,      # Change to 0.3 for 70/30 split, etc.
        n_splits=5          # Change to 10 for 10-fold CV, etc.
    ) 