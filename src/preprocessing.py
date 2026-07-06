import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["Agriculture", "Water", "Urban", "Desert", "Roads", "Trees"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

def main():
    raw_dir = Path("data/raw")
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    
    print("[INFO] Scanning dataset directory structure...")
    
    # Loop over class directories
    for class_name in CLASS_NAMES:
        class_path = raw_dir / class_name
        if not class_path.exists():
            print(f"[WARNING] Directory {class_path} does not exist. Skipping.")
            continue
            
        # Get all files in this class folder
        # We handle both .jpg and .jpeg
        files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg"))
        
        print(f" - Found {len(files)} files for class: {class_name}")
        
        for f in files:
            # We save the relative path to make it portable
            relative_path = f.relative_to(Path("."))
            records.append({
                "filepath": str(relative_path).replace("\\", "/"),  # Normalize path slashes
                "class_name": class_name,
                "label": CLASS_TO_IDX[class_name]
            })

    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print("[ERROR] No files found! Make sure you ran extract_eurosat.py and download_desert.py first.")
        return
        
    print(f"\n[INFO] Total files loaded: {len(df)}")
    print("Class distribution in entire dataset:")
    print(df["class_name"].value_counts().to_string())

    # Train / Val / Test split: 70% / 15% / 15%
    # First, split into train (70%) and temp (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df["label"], 
        random_state=42
    )

    # Next, split temp into validation (15% total) and test (15% total)
    # 0.50 of 30% is 15%
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df["label"], 
        random_state=42
    )

    # Save splits to CSV
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"\n[INFO] Split complete!")
    print(f" - Train samples: {len(train_df)}")
    print(f" - Val samples: {len(val_df)}")
    print(f" - Test samples: {len(test_df)}")
    print(f"[INFO] Saved splits to {output_dir}")

    # Check stratification sanity
    print("\nTrain class distribution:")
    print(train_df["class_name"].value_counts(normalize=True).to_string())
    print("\nVal class distribution:")
    print(val_df["class_name"].value_counts(normalize=True).to_string())
    print("\nTest class distribution:")
    print(test_df["class_name"].value_counts(normalize=True).to_string())

if __name__ == "__main__":
    main()
