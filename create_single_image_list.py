# Create this file: create_single_image_list.py
import os
from pathlib import Path

def create_single_image_list():
    # Read the original query file
    query_file = Path(r'C:\gitRepo\ames\data\roxford5k\test_query.txt')
    output_dir = Path(r'C:\gitRepo\ames\data\roxford5k')
    output_file = output_dir / 'single_image.txt'
    
    # Read first line from test_query.txt
    with open(query_file, 'r') as f:
        first_line = f.readline().strip()
    
    # Extract just the image path (before the comma)
    image_path = first_line.split(',')[0]  # Gets "jpg/all_souls_000013.jpg"
    
    # Write single image to new file
    with open(output_file, 'w') as f:
        f.write(f"{image_path}\n")
    
    print(f"✅ Created '{output_file}' with single image: {image_path}")
    return str(output_file)

if __name__ == '__main__':
    create_single_image_list()