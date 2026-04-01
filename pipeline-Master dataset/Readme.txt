Run: python main.py

Requirement: Segmented images should be stored into the directory "data/segmented_images"

Folder Structure that will create automatically:
data/segmented_images
data/global
data/master
data/local/sand/data
data/local/sand/images
data/local/unreacted/data
data/local/unreacted/images
data/local/porosity/data
data/local/porosity/images
data/filtered_Local/sand
data/filtered_Local/porosity
data/filtered_Local/unreacted

Separate script introduction:
1. Global_properties_extraction.py - The purpose of this file is to extract the phase volume percentage from each image of the folder and save them in an excel file.
2. Sand_local_properties.py - The purpose of this file is to extract the local properties for sand only from the segmented images and save an excel file for each image.
3. Unreacted_local_properties.py - The purpose of this file is to extract the local properties for unreacted particles only from the segmented images and save an excel file for each image.
4. Porosity_local_properties.py - The purpose of this file is to extract the local properties for porosity only from the segmented images and save an excel file for each image.
5. Filter_local_properties.py - The purpose of this file is to filter local properties (For all local files- sand, porosity, unreacted).
6. Master_table_create.py - The purpose of this file  is to convert the local properties into global properties and merge with the main global properties and make a master dataset.


Once the script finish it's runtime you can see the log-file in the root directory.