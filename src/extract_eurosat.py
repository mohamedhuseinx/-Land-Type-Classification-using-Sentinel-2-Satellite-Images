import io
import os
import zipfile
from pathlib import Path
from PIL import Image
import tfrecord

# ── Class Mappings ──────────────────────────────────────────────────
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

EUROSAT_TO_TARGET = {
    "AnnualCrop": "Agriculture",
    "PermanentCrop": "Agriculture",
    "Pasture": "Agriculture",
    "Forest": "Trees",
    "HerbaceousVegetation": "Trees",
    "River": "Water",
    "SeaLake": "Water",
    "Residential": "Urban",
    "Industrial": "Urban",
    "Highway": "Roads"
}

def main():
    # Paths
    tfrecord_path = Path("data/raw/eurosat/rgb/2.0.0/eurosat-train.tfrecord-00000-of-00001")
    output_base_dir = Path("data/raw")

    print("[INFO] Starting EuroSAT extraction...")
    
    # Initialize counts
    counts = {target_class: 0 for target_class in set(EUROSAT_TO_TARGET.values())}
    
    # Ensure target directories exist
    for target_class in counts.keys():
        (output_base_dir / target_class).mkdir(parents=True, exist_ok=True)
    
    # TFRecord loader description
    description = {
        "image": "byte",
        "label": "int",
        "filename": "byte"
    }
    
    # Open TFRecord loader
    loader = tfrecord.reader.tfrecord_loader(str(tfrecord_path), None, description)
    
    total_processed = 0
    
    for record in loader:
        try:
            image_bytes = record["image"]
            label_idx = record["label"][0]
            filename_bytes = record["filename"]
            
            filename = filename_bytes.decode("utf-8") if isinstance(filename_bytes, bytes) else str(filename_bytes)
            
            # EuroSAT class name
            eurosat_class = EUROSAT_CLASSES[label_idx]
            
            # Target class name
            target_class = EUROSAT_TO_TARGET[eurosat_class]
            
            # Target filepath
            # Use the filename from the TFRecord
            output_filepath = output_base_dir / target_class / filename
            
            # Save the image
            # Since the bytes are already in JPEG format, we can write them directly to file!
            with open(output_filepath, "wb") as f:
                f.write(image_bytes)
            
            counts[target_class] += 1
            total_processed += 1
            
            if total_processed % 5000 == 0:
                print(f"[INFO] Processed {total_processed} images...")
                
        except Exception as e:
            print(f"[ERROR] Error processing record: {e}")
            
    print(f"\n[INFO] Extraction complete!")
    print(f"Total processed: {total_processed}")
    print("Class distribution:")
    for target_class, count in counts.items():
        print(f" - {target_class}: {count}")

if __name__ == "__main__":
    main()
