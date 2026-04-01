import cv2
import numpy as np
import os
from openpyxl import Workbook

# --- Folder containing your segmented images ---
input_folder = "data/segmented_images"

# --- Define base colors in BGR format and corresponding phase names ---
color_mapping = {
    (0, 255, 0): "sand",         # Green
    (0, 0, 0): "porosity",       # Black
    (0, 0, 255): "unreacted",    # Red
    (0, 255, 255): "gel",        # Yellow
    (255, 255, 255): "impurity"  # White
}

# --- Create workbook and worksheet ---
wb = Workbook()
ws = wb.active
ws.title = "Phase Percentages"

# Write header row
headers = [
    "serial",
    "image_name",
    "sand_percentage",
    "porosity_percentage",
    "unreacted_percentage",
    "gel_percentage",
    "impurity_percentage"
]
ws.append(headers)

# --- Variables to store averages ---
phase_totals = {phase: 0.0 for phase in color_mapping.values()}
image_count = 0

# --- Loop through images in folder ---
for i, filename in enumerate(sorted(os.listdir(input_folder)), start=1):
    if not (filename.lower().endswith('.png')):
        continue

    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath)
    if image is None:
        print(f"Skipping unreadable file: {filename}")
        continue

    # Flatten and find unique colors
    pixels = image.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    total_pixels = image.shape[0] * image.shape[1]
    percentages = (counts / total_pixels) * 100

    # Initialize each phase percentage to zero
    phase_percentages = {phase: 0.0 for phase in color_mapping.values()}

    # Match detected colors to known phases
    for color, pct in zip(unique_colors, percentages):
        color_tuple = tuple(color)
        if color_tuple in color_mapping:
            phase_name = color_mapping[color_tuple]
            phase_percentages[phase_name] = pct

    # Write one row per image
    ws.append([
        i,
        filename,
        f"{phase_percentages['sand']:.9f}".replace(',', '.'),
        f"{phase_percentages['porosity']:.9f}".replace(',', '.'),
        f"{phase_percentages['unreacted']:.9f}".replace(',', '.'),
        f"{phase_percentages['gel']:.9f}".replace(',', '.'),
        f"{phase_percentages['impurity']:.9f}".replace(',', '.')
    ])

    # Accumulate totals for averaging
    for phase in phase_totals:
        phase_totals[phase] += phase_percentages[phase]

    image_count += 1

# --- Compute and print averages ---
if image_count > 0:
    print("\nAverage percentage across all images:\n")
    for phase, total in phase_totals.items():
        avg = total / image_count
        print(f"{phase.capitalize():<10}: {avg:.2f}%")

# --- Save Excel file ---
output_file = "data/global/global_properties.xlsx"
wb.save(output_file)
print(f"\nData saved successfully to '{output_file}'")
