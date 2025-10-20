import os
import glob

def create_image_list_file(data_path: str, output_filename: str = 'single_image.txt', 
                          image_extensions: list = ['.tif', '.jpg', '.png', '.jpeg']):
    """
    Create image list file for orthophoto dataset
    
    Args:
        data_path: Path to folder containing orthophoto images
        output_filename: Name of output file (default: 'single_image.txt')
        image_extensions: List of image file extensions to include
    """
    
    print(f"🔍 Scanning for images in: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Find all image files
    image_files = []
    
    for ext in image_extensions:
        # Search for files with this extension (case insensitive)
        pattern = os.path.join(data_path, f"*{ext}")
        files = glob.glob(pattern)
        
        # Also try uppercase
        pattern_upper = os.path.join(data_path, f"*{ext.upper()}")
        files.extend(glob.glob(pattern_upper))
        
        image_files.extend(files)
    
    # Remove duplicates and sort
    image_files = list(set(image_files))
    image_files.sort()
    
    print(f"📊 Found {len(image_files)} image files")
    
    if len(image_files) == 0:
        print(f"❌ No image files found in {data_path}")
        print(f"Looking for extensions: {image_extensions}")
        return None
    
    # Create output file path
    output_path = os.path.join(data_path, output_filename)
    
    # Write image list file
    with open(output_path, 'w') as f:
        for i, img_path in enumerate(image_files):
            # Get just the filename
            img_name = os.path.basename(img_path)
            
            # Format: "filename,query_id,width,height"
            # For orthophotos, we'll use standard dimensions and sequential IDs
            line = f"{img_name},{i},1120,700\n"
            f.write(line)
    
    print(f"✅ Image list saved: {output_path}")
    print(f"📄 Format: filename,query_id,width,height")
    
    # Display first few entries as example
    print(f"\n📋 First 5 entries:")
    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"  {line.strip()}")
            else:
                break
    
    return output_path

def verify_orthophoto_filenames(data_path: str):
    """Verify that filenames follow orthophoto naming convention"""
    
    import re
    
    print(f"\n🔍 Verifying orthophoto filename format...")
    
    # Get all .tif files
    tif_files = glob.glob(os.path.join(data_path, "*.tif"))
    tif_files.extend(glob.glob(os.path.join(data_path, "*.TIF")))
    
    valid_count = 0
    invalid_files = []
    
    for file_path in tif_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        # Check if filename matches pattern: imgdb_XXXX_YYYY
        match = re.search(r'(\d+)_(\d+)', base_name)
        
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            print(f"  ✅ {filename} → coordinates ({x}, {y})")
            valid_count += 1
        else:
            print(f"  ❌ {filename} → invalid format (no coordinates found)")
            invalid_files.append(filename)
    
    print(f"\n📊 Verification Results:")
    print(f"  Valid files: {valid_count}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print(f"\n⚠️ Invalid files (won't work with coordinate extraction):")
        for filename in invalid_files:
            print(f"    {filename}")
    
    return valid_count > 0

if __name__ == "__main__":
    # Configuration for your dataset
    data_path = r'C:\OrthoPhoto\Split'
    output_filename = 'single_image.txt'
    
    print(f"🚀 Creating image list for orthophoto dataset")
    print(f"📁 Data path: {data_path}")
    
    # First verify the filenames
    has_valid_files = verify_orthophoto_filenames(data_path)
    
    if not has_valid_files:
        print(f"\n❌ No valid orthophoto files found!")
        print(f"Make sure your files follow the naming pattern: imgdb_XXXX_YYYY.tif")
        exit(1)
    
    # Create the image list file
    output_path = create_image_list_file(
        data_path=data_path,
        output_filename=output_filename,
        image_extensions=['.tif', '.TIF', '.jpg', '.jpeg', '.png']
    )
    
    if output_path:
        print(f"\n🎉 Image list created successfully!")
        print(f"📄 File: {output_path}")
        print(f"\n📝 Next steps:")
        print(f"1. Run create_orthophoto_gnd.py to create ground truth")
        print(f"2. Run extract_descriptors.py to extract features")
        print(f"3. Run your similarity analysis")
    else:
        print(f"\n❌ Failed to create image list file")



     '''
         imgdb_10080_10850.tif,3,1120,700
         imgdb_10080_11200.tif,4,1120,700

        🎉 Image list created successfully!
        📄 File: C:\OrthoPhoto\Split\single_image.txt

        📝 Next steps:
        1. Run create_orthophoto_gnd.py to create ground truth
        2. Run extract_descriptors.py to extract features
        3. Run your similarity analysis
    '''    