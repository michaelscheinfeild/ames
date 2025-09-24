# Create a helper script, e.g., create_file_list.py
import os
from pathlib import Path

def generate_image_list():
    image_dir = Path(r'C:\OrthoPhoto\SmallDb')
    output_dir = Path(r'C:\OrthoPhoto\SmallData\ortho')
    output_file = output_dir / 'image_list.txt'

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files (e.g., .jpg, .png)
    image_files = [f.name for f in image_dir.glob('*.tif')] # Change extension if needed

    # Write the list to the file
    with open(output_file, 'w') as f:
        for fname in image_files:
            f.write(f"{fname}\n")

    print(f"✅ Created '{output_file}' with {len(image_files)} images.")

if __name__ == '__main__':
    generate_image_list()