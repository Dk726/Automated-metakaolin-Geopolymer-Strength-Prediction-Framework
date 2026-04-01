import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color, measure
from skimage.measure import label, regionprops
from openpyxl import Workbook

# --- Paths ---
input_folder = "data/segmented_images"
output_folder_data = "data/local/porosity/data"
output_folder_image = "data/local/porosity/images"

# Create output folders if they don’t exist
os.makedirs(output_folder_data, exist_ok=True)
os.makedirs(output_folder_image, exist_ok=True)

# --- Parameters ---
pixels_to_um = 1  # scale factor (1 px = 0.0827 µm if you wish to adjust)

# --- Process each image ---
for filename in sorted(os.listdir(input_folder)):
    if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
        continue

    # Read image
    filepath = os.path.join(input_folder, filename)
    img = cv2.imread(filepath, 1)
    if img is None:
        print(f"⚠️ Skipping unreadable file: {filename}")
        continue
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width, channels = img_rgb.shape

    # Create a new white image (background)
    # White in BGR is (255, 255, 255)
    new_img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Define lower and upper bounds for "black" pixels
    # Exact black:
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([0, 0, 0], dtype=np.uint8)

    # Create a mask for black pixels
    mask = cv2.inRange(img_rgb, lower_black, upper_black)

    # Copy black pixels from original image to new image
    new_img[mask == 255] = (0, 255, 0)
    
    # Define lower/upper bounds for green (sand)
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)

    # Create mask and extract only green regions
    mask_green = cv2.inRange(new_img, lower_green, upper_green)
    green_only = cv2.bitwise_and(new_img, new_img, mask=mask_green)
    
    black_only_gray = cv2.cvtColor(green_only, cv2.COLOR_RGB2GRAY)
    plt.imshow(black_only_gray)

    # Label connected regions
    s = np.ones((3, 3), dtype=int)
    label_mask, num_labels = ndimage.label(black_only_gray, structure=s)

    # Extract region properties
    clusters = measure.regionprops(label_mask, img)
    propertyList = [
        'centroid',
        'Area',
        'equivalent_diameter',
        'orientation',
        'MajorAxisLength',
        'MinorAxisLength',
        'Perimeter'
    ]

    # --- Create and write Excel file ---
    wb = Workbook()
    ws = wb.active
    ws.title = "Porosity Measurements"
    ws.append(['Label'] + propertyList)

    for cluster_props in clusters:
        row_data = [cluster_props['Label']]
        for prop in propertyList:
            if prop == 'Area':
                to_print = cluster_props[prop] * pixels_to_um**2
            elif prop == 'centroid':
                y, x = cluster_props[prop]
                to_print = f"({y:.2f}, {x:.2f})"
            elif prop == 'orientation':
                to_print = cluster_props[prop] * 57.2958  # radians to degrees
            elif 'Intensity' not in prop:
                to_print = cluster_props[prop] * pixels_to_um
            else:
                to_print = cluster_props[prop]
            if isinstance(to_print, (int, float)):
                to_print = f"{to_print:.6f}".replace(',', '.')
            row_data.append(to_print)
        ws.append(row_data)

    image_name_no_ext = os.path.splitext(filename)[0]
    excel_output_path = os.path.join(output_folder_data, f"porosity_property_{image_name_no_ext}.xlsx")
    wb.save(excel_output_path)
    print(f"Saved Excel: porosity_property_{image_name_no_ext}.xlsx")
"""
    # --- Generate marked image ---
    regions = regionprops(label_mask)
    fig, ax = plt.subplots()
    ax.imshow(black_only_gray, cmap=plt.cm.gray)

    for i, props in enumerate(regions, start=1):
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=0.5)
        ax.plot(x0, y0, '.g', markersize=5)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=0.5)

        # Serial number label
        ax.text(x0 + 10, y0, str(i), color='yellow', fontsize=8, weight='normal')

    ax.axis((0, 1536, 1024, 0))
    plt.tight_layout()

    # Save marked image
    image_output_path = os.path.join(output_folder_image, f"{image_name_no_ext}_marked.png")
    plt.savefig(image_output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved marked image: {image_name_no_ext}_marked.png")
"""
print("\nAll images processed, Excel files and marked images saved successfully!")
