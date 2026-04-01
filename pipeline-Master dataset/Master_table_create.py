import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def process_local_properties(local_df, phase_name):
    """Calculate statistics from local properties dataframe for a specific phase"""
    
    # Initialize results dictionary with NaN values
    results = {
        f'{phase_name}_particles_count': np.nan,
        f'{phase_name}_area_mean': np.nan,
        f'{phase_name}_area_standard_deviation': np.nan,
        f'{phase_name}_area_min_value': np.nan,
        f'{phase_name}_area_max_value': np.nan,
        f'{phase_name}_area_median': np.nan,
        f'{phase_name}_equivalent_diameter_mean': np.nan,
        f'{phase_name}_equivalent_diameter_standard_deviation': np.nan,
        f'{phase_name}_equivalent_diameter_min_value': np.nan,
        f'{phase_name}_equivalent_diameter_max_value': np.nan,
        f'{phase_name}_equivalent_diameter_median': np.nan,
        f'{phase_name}_major_axis_mean': np.nan,
        f'{phase_name}_major_axis_standard_deviation': np.nan,
        f'{phase_name}_major_axis_min_value': np.nan,
        f'{phase_name}_major_axis_max_value': np.nan,
        f'{phase_name}_major_axis_median': np.nan,
        f'{phase_name}_minor_axis_mean': np.nan,
        f'{phase_name}_minor_axis_standard_deviation': np.nan,
        f'{phase_name}_minor_axis_min_value': np.nan,
        f'{phase_name}_minor_axis_max_value': np.nan,
        f'{phase_name}_minor_axis_median': np.nan,
        f'{phase_name}_aspect_ratio_mean': np.nan,
        f'{phase_name}_aspect_ratio_standard_deviation': np.nan,
        f'{phase_name}_aspect_ratio_min_value': np.nan,
        f'{phase_name}_aspect_ratio_max_value': np.nan,
        f'{phase_name}_aspect_ratio_median': np.nan,
        f'{phase_name}_perimeter_mean': np.nan,
        f'{phase_name}_perimeter_standard_deviation': np.nan,
        f'{phase_name}_perimeter_min_value': np.nan,
        f'{phase_name}_perimeter_max_value': np.nan,
        f'{phase_name}_perimeter_median': np.nan,
        f'{phase_name}_orientation_mean': np.nan,
    }
    
    # Check if dataframe is empty
    if local_df.empty:
        return results
    
    # Calculate aspect ratio
    if 'MajorAxisLength' in local_df.columns and 'MinorAxisLength' in local_df.columns:
        local_df['AspectRatio'] = local_df['MajorAxisLength'] / local_df['MinorAxisLength']
    
    # 1. Particle count
    results[f'{phase_name}_particles_count'] = len(local_df)
    
    # 2-6. Area statistics
    if 'Area' in local_df.columns:
        results[f'{phase_name}_area_mean'] = local_df['Area'].mean()
        results[f'{phase_name}_area_standard_deviation'] = local_df['Area'].std()
        results[f'{phase_name}_area_min_value'] = local_df['Area'].min()
        results[f'{phase_name}_area_max_value'] = local_df['Area'].max()
        results[f'{phase_name}_area_median'] = local_df['Area'].median()
    
    # 7-11. Equivalent diameter statistics
    if 'equivalent_diameter' in local_df.columns:
        results[f'{phase_name}_equivalent_diameter_mean'] = local_df['equivalent_diameter'].mean()
        results[f'{phase_name}_equivalent_diameter_standard_deviation'] = local_df['equivalent_diameter'].std()
        results[f'{phase_name}_equivalent_diameter_min_value'] = local_df['equivalent_diameter'].min()
        results[f'{phase_name}_equivalent_diameter_max_value'] = local_df['equivalent_diameter'].max()
        results[f'{phase_name}_equivalent_diameter_median'] = local_df['equivalent_diameter'].median()
    
    # 12-16. Major axis statistics
    if 'MajorAxisLength' in local_df.columns:
        results[f'{phase_name}_major_axis_mean'] = local_df['MajorAxisLength'].mean()
        results[f'{phase_name}_major_axis_standard_deviation'] = local_df['MajorAxisLength'].std()
        results[f'{phase_name}_major_axis_min_value'] = local_df['MajorAxisLength'].min()
        results[f'{phase_name}_major_axis_max_value'] = local_df['MajorAxisLength'].max()
        results[f'{phase_name}_major_axis_median'] = local_df['MajorAxisLength'].median()
    
    # 17-21. Minor axis statistics
    if 'MinorAxisLength' in local_df.columns:
        results[f'{phase_name}_minor_axis_mean'] = local_df['MinorAxisLength'].mean()
        results[f'{phase_name}_minor_axis_standard_deviation'] = local_df['MinorAxisLength'].std()
        results[f'{phase_name}_minor_axis_min_value'] = local_df['MinorAxisLength'].min()
        results[f'{phase_name}_minor_axis_max_value'] = local_df['MinorAxisLength'].max()
        results[f'{phase_name}_minor_axis_median'] = local_df['MinorAxisLength'].median()
    
    # 22-26. Aspect ratio statistics
    if 'AspectRatio' in local_df.columns:
        results[f'{phase_name}_aspect_ratio_mean'] = local_df['AspectRatio'].mean()
        results[f'{phase_name}_aspect_ratio_standard_deviation'] = local_df['AspectRatio'].std()
        results[f'{phase_name}_aspect_ratio_min_value'] = local_df['AspectRatio'].min()
        results[f'{phase_name}_aspect_ratio_max_value'] = local_df['AspectRatio'].max()
        results[f'{phase_name}_aspect_ratio_median'] = local_df['AspectRatio'].median()
    
    # 27-31. Perimeter statistics
    if 'Perimeter' in local_df.columns:
        results[f'{phase_name}_perimeter_mean'] = local_df['Perimeter'].mean()
        results[f'{phase_name}_perimeter_standard_deviation'] = local_df['Perimeter'].std()
        results[f'{phase_name}_perimeter_min_value'] = local_df['Perimeter'].min()
        results[f'{phase_name}_perimeter_max_value'] = local_df['Perimeter'].max()
        results[f'{phase_name}_perimeter_median'] = local_df['Perimeter'].median()
        
    # 32. Orientation statistics
    if 'orientation' in local_df.columns:
        results[f'{phase_name}_orientation_mean'] = local_df['orientation'].mean()
    
    return results

def generate_column_names(phase_name):
    """Generate column names for a specific phase"""
    return [
        f'{phase_name}_particles_count',
        f'{phase_name}_area_mean', f'{phase_name}_area_standard_deviation', 
        f'{phase_name}_area_min_value', f'{phase_name}_area_max_value', f'{phase_name}_area_median',
        f'{phase_name}_equivalent_diameter_mean', f'{phase_name}_equivalent_diameter_standard_deviation',
        f'{phase_name}_equivalent_diameter_min_value', f'{phase_name}_equivalent_diameter_max_value',
        f'{phase_name}_equivalent_diameter_median',
        f'{phase_name}_major_axis_mean', f'{phase_name}_major_axis_standard_deviation',
        f'{phase_name}_major_axis_min_value', f'{phase_name}_major_axis_max_value', f'{phase_name}_major_axis_median',
        f'{phase_name}_minor_axis_mean', f'{phase_name}_minor_axis_standard_deviation',
        f'{phase_name}_minor_axis_min_value', f'{phase_name}_minor_axis_max_value', f'{phase_name}_minor_axis_median',
        f'{phase_name}_aspect_ratio_mean', f'{phase_name}_aspect_ratio_standard_deviation',
        f'{phase_name}_aspect_ratio_min_value', f'{phase_name}_aspect_ratio_max_value', f'{phase_name}_aspect_ratio_median',
        f'{phase_name}_perimeter_mean', f'{phase_name}_perimeter_standard_deviation',
        f'{phase_name}_perimeter_min_value', f'{phase_name}_perimeter_max_value', f'{phase_name}_perimeter_median', f'{phase_name}_orientation_mean'
    ]

def main():
    # Define paths
    base_path = Path("data/")
    global_properties_path = base_path / "global" / "global_properties.xlsx"
    
    # Define phases and their corresponding folders
    phases = {
        'sand': base_path / "filtered_Local" / "sand",
        'porosity': base_path / "filtered_Local" / "porosity",
        'unreacted': base_path / "filtered_Local" / "unreacted"
    }
    
    # Create output path
    output_path = base_path / "master" / "master.xlsx"
    
    # Check if input files exist
    if not global_properties_path.exists():
        print(f"Error: Global properties file not found at {global_properties_path}")
        return
    
    # Check if all phase directories exist
    for phase_name, phase_path in phases.items():
        if not phase_path.exists():
            print(f"Warning: {phase_name} directory not found at {phase_path}")
            print("  Statistics for this phase will be filled with NaN values.")
    
    # Read global properties
    print(f"Reading global properties from {global_properties_path}")
    global_df = pd.read_excel(global_properties_path)
    
    # Generate all column names for all phases
    all_columns = []
    for phase_name in phases.keys():
        all_columns.extend(generate_column_names(phase_name))
    
    # Initialize lists for new columns
    new_columns_data = {col: [] for col in all_columns}
    
    # Process each row
    total_rows = len(global_df)
    for idx, row in global_df.iterrows():
        image_name = row['image_name']
        
        # Remove extension and get image base name
        image_base = os.path.splitext(image_name)[0]  # Removes .png
        
        # Show progress
        print(f"Processing {idx+1}/{total_rows}: {image_name}")
        
        # Initialize dictionary to hold all stats for this image
        all_stats = {}
        
        # Process each phase
        for phase_name, phase_path in phases.items():
            # Construct local filename
            local_filename = f"{phase_name}_property_{image_base}.xlsx"
            local_filepath = phase_path / local_filename
            
            if local_filepath.exists():
                # Read local properties
                try:
                    local_df = pd.read_excel(local_filepath)
                    # Calculate statistics for this phase
                    phase_stats = process_local_properties(local_df, phase_name)
                    all_stats.update(phase_stats)
                    
                except Exception as e:
                    print(f"  Error processing {phase_name} file {local_filename}: {e}")
                    # Fill with NaN for all columns for this phase
                    phase_columns = generate_column_names(phase_name)
                    for col in phase_columns:
                        all_stats[col] = np.nan
            else:
                # Fill with NaN for all columns for this phase
                phase_columns = generate_column_names(phase_name)
                for col in phase_columns:
                    all_stats[col] = np.nan
        
        # Append statistics to lists for all columns
        for col in all_columns:
            new_columns_data[col].append(all_stats.get(col, np.nan))
    
    # Add new columns to global dataframe
    for col_name, col_data in new_columns_data.items():
        global_df[col_name] = col_data
    
    # Save to new Excel file
    print(f"\nSaving combined properties to {output_path}")
    global_df.to_excel(output_path, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Total rows processed: {total_rows}")
    print(f"Output saved to: {output_path}")
    
    # Show summary
    print("\nSample of combined data (first few columns):")
    print(global_df.head())
    
    # Print column summary
    print("\n\nColumn summary:")
    print(f"Total columns in output: {len(global_df.columns)}")
    print(f"Original global columns: {len(global_df.columns) - len(all_columns)}")
    print(f"New local property columns: {len(all_columns)}")
    
    # Show breakdown by phase
    print("\nColumns added per phase:")
    for phase_name in phases.keys():
        phase_cols = [col for col in global_df.columns if col.startswith(f'{phase_name}_')]
        print(f"  {phase_name}: {len(phase_cols)} columns")
    
    return global_df

if __name__ == "__main__":
    # Run the main function
    combined_df = main()