'''
after extract printed 
🎉 EXTRACTION COMPLETE!
✅ Output file: C:\gitRepo\ames\data/roxford5k/dinov2_query_local.hdf5
📊 Expected shape: (1, 600, 773)
📋 Data format: [1 image, 600 patches, 5 metadata + 768 features]

'''


'''
Result
✅ File loaded successfully!
📊 Shape: (1, 600, 773)
📋 Data type: float32
🔢 Min value: -0.1561
🔢 Max value: 194.6441

📍 Sample metadata (first 10 patches):
Coordinates (x,y): [[0.7585425  0.24005753]
 [0.44541115 0.67789537]
 [0.44541115 0.6049224 ]]
Masks: [0. 0. 0. 0. 0.]

🧠 Sample features (first patch):
Feature vector shape: (768,)
Feature range: [-0.1010, 0.1114]



big data if copy
📊 Shape: (70,)
📋 Data type: [('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))]

'''
# verify_extraction.py


# verify_extraction.py
import h5py
import numpy as np

def verify_global_format(global_file=None):
    """Inspect the global features file structure"""
  
    
    try:
        with h5py.File(global_file, 'r') as f:
            features = f['features'][:]
            print(f"✅ Global file loaded successfully!")
            print(f"📊 Shape: {features.shape}")
            print(f"📋 Data type: {features.dtype}")
            print(f"🔢 Min value: {features.min():.4f}")
            print(f"🔢 Max value: {features.max():.4f}")
            
            # Check if it's a single image (should be shape (1, 768))
            if len(features.shape) == 2 and features.shape[0] == 1:
                print(f"\n🎯 GLOBAL FEATURES ANALYSIS:")
                print(f"Number of images: {features.shape[0]}")
                print(f"Feature dimensions: {features.shape[1]} (should be 768 for DINOv2)")

                               # Show sample values
                global_vector = features[0]  # Single global vector
                print(f"\n📊 Global vector statistics:")
                print(f"Vector shape: {global_vector.shape}")
                print(f"L2 norm: {np.linalg.norm(global_vector):.4f}")
                print(f"Mean value: {global_vector.mean():.4f}")
                print(f"Std deviation: {global_vector.std():.4f}")
                print(f"First 10 values: {global_vector[:10]}")
                print(f"Last 10 values: {global_vector[-10:]}")
                
                # Check if normalized (L2 norm should be ~1.0)
                norm = np.linalg.norm(global_vector)
                is_normalized = abs(norm - 1.0) < 1e-5
                print(f"Is L2 normalized: {'✅' if is_normalized else '❌'} (norm = {norm:.6f})")
                
                # Check for any zero values
                zero_count = np.sum(global_vector == 0.0)
                print(f"Zero values: {zero_count} / {len(global_vector)}")
                
            else:
                print(f"⚠️ Unexpected shape for global features: {features.shape}")
                print(f"Expected: (1, 768) for single image")
                
    except Exception as e:
        print(f"❌ Error loading global file: {e}")
                

def compare_flat_vs_structured():
    """Compare single flat format data with first entry in structured format"""
    
    # File paths
    flat_file = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
    structured_file = r'C:\gitRepo\ames\data\roxford5k-Copy\dinov2_query_local.hdf5'
    
    try:
        # Load flat format data
        with h5py.File(flat_file, 'r') as f:
            flat_data = f['features'][0]  # Shape: (600, 773)
            flat_metadata = flat_data[:, :5]      # (600, 5)
            flat_descriptors = flat_data[:, 5:]   # (600, 768)
        
        # Load structured format data (first entry)
        with h5py.File(structured_file, 'r') as f:
            struct_data = f['features'][0]  # First image from structured array
            struct_metadata = struct_data['metadata']    # (700, 5)
            struct_descriptors = struct_data['descriptor'] # (700, 768)
        
        print("🔍 FORMAT COMPARISON - Single Image Data")
        print("="*60)
        
        # Compare shapes
        print(f"📊 SHAPES:")
        print(f"Flat metadata:       {flat_metadata.shape}")
        print(f"Structured metadata: {struct_metadata.shape}")
        print(f"Flat descriptors:    {flat_descriptors.shape}")
        print(f"Structured descriptors: {struct_descriptors.shape}")
        
        # Compare metadata (first 5 columns)
        print(f"\n🎯 METADATA COMPARISON (First 10 patches):")
        print(f"{'Column':<12} {'Flat':<20} {'Structured':<20} {'Match':<8}")
        print("-" * 65)
        
        for i in range(5):  # 5 metadata columns
            col_names = ['X-coord', 'Y-coord', 'Scale', 'Mask', 'Weight']
            flat_sample = flat_metadata[:10, i]
            struct_sample = struct_metadata[:10, i]
            
            # Check if values are close (for floating point comparison)
            close_match = np.allclose(flat_sample, struct_sample, rtol=1e-5)
            match_status = "✅" if close_match else "❌"
            
            print(f"{col_names[i]:<12} {str(flat_sample[:3]):<20} {str(struct_sample[:3]):<20} {match_status:<8}")
        
        # Compare descriptors (first few features)
        print(f"\n🧠 DESCRIPTOR COMPARISON (First patch, first 10 features):")
        flat_feat_sample = flat_descriptors[0, :10]
        struct_feat_sample = struct_descriptors[0, :10]
        feat_match = np.allclose(flat_feat_sample, struct_feat_sample, rtol=1e-3)
        
        print(f"Flat features:       {flat_feat_sample}")
        print(f"Structured features: {struct_feat_sample}")
        print(f"Features match:      {'✅' if feat_match else '❌'}")
        
        # Mask interpretation analysis
        print(f"\n🎭 MASK INTERPRETATION:")
        flat_masks = flat_metadata[:, 3]
        struct_masks = struct_metadata[:, 3]
        
        print(f"Flat format - Valid patches (mask=0.0):      {np.sum(flat_masks == 0.0)}")
        print(f"Flat format - Invalid patches (mask=1.0):    {np.sum(flat_masks == 1.0)}")
        print(f"Structured - Valid patches (mask=1.0):       {np.sum(struct_masks == 1.0)}")
        print(f"Structured - Invalid patches (mask=0.0):     {np.sum(struct_masks == 0.0)}")
        
        # Check if masks are inverted
        inverted_match = np.allclose(flat_masks, 1.0 - struct_masks[:600])
        print(f"Masks are inverted: {'✅' if inverted_match else '❌'}")
        
        # Data type comparison
        print(f"\n🔧 DATA TYPES:")
        print(f"Flat metadata dtype:       {flat_metadata.dtype}")
        print(f"Structured metadata dtype: {struct_metadata.dtype}")
        print(f"Flat descriptors dtype:    {flat_descriptors.dtype}")
        print(f"Structured descriptors dtype: {struct_descriptors.dtype}")
        
        return {
            'shapes_match': flat_descriptors.shape[1] == struct_descriptors.shape[1],
            'metadata_close': np.allclose(flat_metadata[:, :2], struct_metadata[:600, :2], rtol=1e-5),
            'masks_inverted': inverted_match,
            'features_close': np.allclose(flat_descriptors, struct_descriptors[:600].astype(np.float32), rtol=1e-3)
        }
        
    except Exception as e:
        print(f"❌ Error comparing formats: {e}")
        return None




def verify_structured_format():
    """Handle structured array format with named fields"""
    file_path = r'C:\gitRepo\ames\data\roxford5k-Copy\dinov2_query_local.hdf5'
    
    try:
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            print(f"✅ File loaded successfully!")
            print(f"📊 Shape: {features.shape}")
            print(f"📋 Data type: {features.dtype}")
            
            # For structured arrays, access named fields:
            metadata = features['metadata']  # Shape: (70, 700, 5)
            descriptors = features['descriptor']  # Shape: (70, 700, 768)
            
            print(f"\n📊 STRUCTURED ARRAY FORMAT:")
            print(f"Metadata shape: {metadata.shape}")
            print(f"Descriptor shape: {descriptors.shape}")
            print(f"Metadata dtype: {metadata.dtype}")
            print(f"Descriptor dtype: {descriptors.dtype}")
            
            # Get masks from metadata (index 3)
            # metadata structure: [x, y, scale, mask, weight] 
            #                     [0, 1,   2,    3,    4   ]
            
            # Check masks for first image
            first_image_metadata = metadata[0]  # Shape: (700, 5)
            masks = first_image_metadata[:, 3]  # Extract mask column
            
            print(f"\n🎭 MASK ANALYSIS (First Image):")
            print(f"Total patches: {len(masks)}")
            print(f"Valid patches (mask=1.0): {np.sum(masks == 1.0)}")
            print(f"Invalid/padding patches (mask=0.0): {np.sum(masks == 0.0)}")
            print(f"Unique mask values: {np.unique(masks)}")
            print(f"First 20 masks: {masks[:20]}")
            print(f"Last 20 masks: {masks[-20:]}")
            
            # Coordinates analysis
            coords = first_image_metadata[:, :2]  # x, y coordinates
            print(f"\n📍 COORDINATE ANALYSIS:")
            print(f"X range: [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
            print(f"Y range: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
            print(f"Sample coordinates (first 5): \n{coords[:5]}")
            
            # Check all images' masks
            print(f"\n🌍 ALL IMAGES MASK SUMMARY:")
            for i in range(min(10, len(metadata))):  # Check first 10 images
                img_masks = metadata[i][:, 3]
                valid_count = np.sum(img_masks == 1.0)
                invalid_count = np.sum(img_masks == 0.0)
                print(f"Image {i}: {valid_count} valid, {invalid_count} padding patches")
                
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def verify_flat_format():
    """Handle flat array format (your single image extraction)"""
    file_path = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
    
    try:
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            print(f"✅ File loaded successfully!")
            print(f"📊 Shape: {features.shape}")
            print(f"📋 Data type: {features.dtype}")
            
            # For flat arrays: [batch, patches, metadata+features]
            # Structure: [x, y, scale, mask, weight, feat0, feat1, ..., feat767]
            #            [0, 1,   2,    3,    4,     5,    6,        772]
            
            metadata = features[..., :5]  # First 5 columns
            descriptors = features[..., 5:]  # Last 768 columns
            
            print(f"\n📊 FLAT ARRAY FORMAT:")
            print(f"Metadata shape: {metadata.shape}")
            print(f"Descriptor shape: {descriptors.shape}")
            
            # Extract masks (column 3)
            masks = features[0, :, 3]  # All masks for first (and only) image
            
            print(f"\n🎭 MASK ANALYSIS:")
            print(f"Total patches: {len(masks)}")
            print(f"Valid patches (mask=0.0): {np.sum(masks == 0.0)}")
            print(f"Invalid/padding patches (mask=1.0): {np.sum(masks == 1.0)}")
            print(f"Unique mask values: {np.unique(masks)}")
            print(f"All masks: {masks}")
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")

def auto_detect_and_verify(file_path):
    """Automatically detect format and verify accordingly"""
    try:
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            print(f"📁 File: {file_path}")
            print(f"📊 Shape: {features.shape}")
            print(f"📋 Data type: {features.dtype}")
            
            # Check if structured array
            if features.dtype.names is not None:
                print(f"🔧 Detected: STRUCTURED ARRAY (Format 2)")
                print(f"Fields: {features.dtype.names}")
                
                # Extract metadata and get masks
                metadata = features['metadata']
                masks = metadata[0, :, 3]  # First image, mask column
                
            else:
                print(f"🔧 Detected: FLAT ARRAY (Format 1)")
                
                # Extract masks directly
                masks = features[0, :, 3]  # First image, mask column
            
            # Analyze masks regardless of format
            print(f"\n🎭 MASK ANALYSIS:")
            print(f"Total patches: {len(masks)}")
            print(f"Valid patches (mask=1.0): {np.sum(masks == 1.0)}")
            print(f"Valid patches (mask=0.0): {np.sum(masks == 0.0)}")
            print(f"Unique values: {np.unique(masks)}")
            
            return masks
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':

    global_file = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_global.hdf5'
    print("=== GLOBAL FEATURES INSPECTION ===")
    verify_global_format(global_file)
    '''
    📊 Global vector statistics:
    Vector shape: (768,)
    First 10 values: [-0.00151437  0.02162817 -0.00580567  0.03903817 -0.0016877  -0.02010243
    -0.01618879 -0.02657635  0.02838697  0.05757562]
    '''
    if 0:
        results = compare_flat_vs_structured()
        if results:
            print(f"\n📋 SUMMARY:")
            for key, value in results.items():
                status = "✅" if value else "❌"
                print(f"{key}: {status}")

        print("=== STRUCTURED FORMAT (Big Data) ===")
        verify_structured_format()
    
    print("\n" + "="*50)
    print("=== FLAT FORMAT (Single Image) ===")
    verify_flat_format()
    
    if 0:
        print("\n" + "="*50)
        print("=== AUTO-DETECT FORMAT ===")
        auto_detect_and_verify(r'C:\gitRepo\ames\data\roxford5k-Copy\dinov2_query_local.hdf5')





'''
import h5py
import numpy as np

# for single query
def verify_extraction():
    #file_path = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
    file_path = r'C:\gitRepo\ames\data\roxford5k-Copy\dinov2_query_local.hdf5'
    
    
    try:
        with h5py.File(file_path, 'r') as f:
            features = f['features'][:]
            print(f"✅ File loaded successfully!")
            print(f"📊 Shape: {features.shape}")
            print(f"📋 Data type: {features.dtype}")
            #print(f"🔢 Min value: {features.min():.4f}")
            #print(f"🔢 Max value: {features.max():.4f}")
          
            
            # Check metadata (first 5 columns)
            metadata = features[0, :10, :5]  # First 10 patches, metadata only
            print(f"\n📍 Sample metadata (first 10 patches):")
            print(f"Coordinates (x,y): {metadata[:3, :2]}")
            print(f"Masks: {metadata[:5, 3]}")  # Should be all 1.0 for valid patches
            
            # Check features (last 768 columns)
            feature_data = features[0, 0, 5:]  # First patch, feature vector
            print(f"\n🧠 Sample features (first patch):")
            print(f"Feature vector shape: {feature_data.shape}")
            print(f"Feature range: [{feature_data.min():.4f}, {feature_data.max():.4f}]")


            # stem all masks
            all_masks = features[0, :, 3]  # All masks for the first image
            print(f"\n🎯 All masks for the first image (should be 0.0 for invalid patches):")
            #print(all_masks)
            print(f"Total patches: {len(all_masks)}, Valid patches (mask=0.0): {np.sum(all_masks == 0.0)}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == '__main__':
    verify_extraction()


'''    

'''
=== FLAT FORMAT (Single Image) ===
✅ File loaded successfully!
📊 Shape: (1, 600, 773)
📋 Data type: float32

📊 FLAT ARRAY FORMAT:
Metadata shape: (1, 600, 5)
Descriptor shape: (1, 600, 768)


📊 Shape: (70,)
📋 Data type: [('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))]
🔧 Detected: STRUCTURED ARRAY (Format 2)
Fields: ('metadata', 'descriptor')

'''