import os
import pandas as pd
import numpy as np

# Base directories
base_src_dir = r"data/local"
base_dst_dir = r"data/filtered_Local"

# List of data types to process
data_types = ['sand', 'porosity', 'unreacted']

# Process each data type
for data_type in data_types:
    # Construct source and destination paths
    src_dir = os.path.join(base_src_dir, data_type, 'data')
    dst_dir = os.path.join(base_dst_dir, data_type)
    
    # Check if source directory exists
    if not os.path.exists(src_dir):
        print(f"Source directory not found: {src_dir}")
        continue
    
    # Create destination directory if it does not exist
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"\nProcessing {data_type} data...")
    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    
    # Loop through all .xlsx files in the source folder
    processed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(".xlsx"):
            src_path = os.path.join(src_dir, filename)

            try:
                # Read the Excel file
                df = pd.read_excel(src_path)
                
                # Ensure required columns exist
                required_cols = ["MajorAxisLength", "orientation"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    print(f"  Skipping {filename}: missing columns {missing}")
                    skipped_count += 1
                    continue
                
                # Select numeric columns
                numeric_cols = df.select_dtypes(include="number").columns
                
                # Replace exact zeros with epsilon
                eps = 1e-3  # 0.001
                df[numeric_cols] = df[numeric_cols].replace(0, eps)
                
                # Fix Orientation: if negative, add 180
                df.loc[df["orientation"] < 0, "orientation"] += 180
                
                # Filter rows where MajorAxisLength > 2
                df = df[df["MajorAxisLength"] > 2]
                
                # Save to destination folder with same filename
                dst_path = os.path.join(dst_dir, filename)
                df.to_excel(dst_path, index=False)
                
                processed_count += 1
                print(f"  Processed and saved: {filename}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")
                skipped_count += 1
    
    print(f"Summary: {processed_count} files processed, {skipped_count} files skipped")

print("\nAll data types processed successfully!")