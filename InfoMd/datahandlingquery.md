# Data Handling for Query Processing in AMES

## 📋 Overview

This document covers data format compatibility, conversion methods, and best practices for handling query data in the AMES (Adaptive Multi-scale Enhancement for Similarity) image retrieval pipeline.

## 🔍 Format Compatibility Analysis

### Key Question: Can Flat Format Work with Structured Format?

**Answer:** Limited compatibility with potential critical issues.

## 📊 Data Format Comparison

### Flat Format (Single Descriptor Type)
```python
# Shape: [N_images, topk_patches, 773]
# Structure: [x, y, scale, mask, weight, feature_1, ..., feature_768]
# Dtype: float32 throughout
# Mask convention: 0.0 = valid patch, 1.0 = invalid/padding

# Example access:
with h5py.File('flat_query.hdf5', 'r') as f:
    data = f['features'][:]  # Shape: (1, 600, 773)
    metadata = data[0, :, :5]        # [x, y, scale, mask, weight]
    descriptors = data[0, :, 5:]     # 768-dim DINOv2 features
    valid_mask = metadata[:, 3] == 0.0  # Find valid patches
```

### Structured Format (Multiple Descriptor Types)
```python
# Dtype: [('metadata', '<f4', (600, 5)), ('descriptor', '<f2', (600, 768))]
# Structure: Named fields with different precisions
# Mask convention: 1.0 = valid patch, 0.0 = invalid/padding

# Example access:
with h5py.File('structured_query.hdf5', 'r') as f:
    data = f['features'][:]  # Shape: (1,) with structured dtype
    metadata = data['metadata'][0]      # Shape: (600, 5), dtype: float32
    descriptors = data['descriptor'][0] # Shape: (600, 768), dtype: float16
    valid_mask = metadata[:, 3] == 1.0  # Find valid patches
```

## ⚠️ Critical Compatibility Issues

### 1. Data Access Pattern Mismatch
```python
# AMES expecting STRUCTURED format:
query_features = query_data['descriptor'][0]    # ❌ Fails on flat format
query_metadata = query_data['metadata'][0]     # ❌ Fails on flat format

# AMES expecting FLAT format:
query_features = query_data[0, :, 5:]          # ❌ Fails on structured format
query_metadata = query_data[0, :, :5]         # ❌ Fails on structured format
```

### 2. Mask Interpretation (CRITICAL!)
```python
# STRUCTURED format convention:
valid_patches = metadata[:, 3] == 1.0  # 1.0 = valid patch

# FLAT format convention:  
valid_patches = metadata[:, 3] == 0.0  # 0.0 = valid patch

# Wrong interpretation results in:
# - Using only padding patches for matching (0% accuracy)
# - Skipping all real content patches
```

### 3. Data Type Precision Differences
```python
# STRUCTURED: descriptors are float16 (memory efficient)
descriptors = features['descriptor'][0]  # dtype: float16

# FLAT: descriptors are float32 (full precision)  
descriptors = features[0, :, 5:]        # dtype: float32

# May cause subtle numerical differences in similarity calculations
```

## 🛠️ Conversion Solutions

### Option 1: Convert Flat to Structured Format

```python
# convert_flat_to_structured.py
import h5py
import numpy as np

def convert_flat_to_structured(flat_file, output_file, topk=600):
    """
    Convert flat format HDF5 file to structured format
    
    Args:
        flat_file: Path to flat format HDF5 file
        output_file: Path for output structured format file
        topk: Number of patches per image (default: 600)
    """
    
    print(f"🔄 Converting {flat_file} to structured format...")
    
    # Load flat format data
    with h5py.File(flat_file, 'r') as f:
        flat_data = f['features'][:]
        print(f"📥 Loaded flat data shape: {flat_data.shape}")
        print(f"📥 Data type: {flat_data.dtype}")
    
    # Extract components
    metadata = flat_data[..., :5]    # [N, 600, 5] - [x, y, scale, mask, weight]
    descriptors = flat_data[..., 5:] # [N, 600, 768] - DINOv2 features
    
    # CRITICAL: Fix mask convention (0.0 -> 1.0 for valid patches)
    print("🔧 Converting mask convention: 0.0->1.0 for valid patches")
    metadata[..., 3] = 1.0 - metadata[..., 3]  # Flip mask values
    
    # Verify mask conversion
    n_valid_before = np.sum(flat_data[..., 3] == 0.0)
    n_valid_after = np.sum(metadata[..., 3] == 1.0)
    print(f"✅ Valid patches: {n_valid_before} -> {n_valid_after}")
    
    # Create structured array
    n_images = flat_data.shape[0]
    dt = np.dtype([
        ('metadata', np.float32, (topk, 5)),
        ('descriptor', np.float16, (topk, 768))
    ])
    
    structured_data = np.zeros(n_images, dtype=dt)
    structured_data['metadata'] = metadata.astype(np.float32)
    structured_data['descriptor'] = descriptors.astype(np.float16)
    
    # Save structured format
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=structured_data)
    
    print(f"✅ Converted to structured format: {output_file}")
    print(f"📊 Output shape: {structured_data.shape}")
    print(f"📊 Output dtype: {structured_data.dtype}")
    
    return output_file

# Usage example:
if __name__ == "__main__":
    convert_flat_to_structured(
        'C:/gitRepo/ames/data/roxford5k/dinov2_query_local.hdf5',
        'C:/gitRepo/ames/data/roxford5k/dinov2_query_local_structured.hdf5'
    )
```

### Option 2: Re-extract with Structured Format

```python
# In your extraction script, ensure structured format creation:
desc_type = 'local,global'  # Multiple types trigger structured format

# Verify the format after extraction:
def verify_extraction_format(hdf5_file):
    """Verify if extraction created structured format"""
    with h5py.File(hdf5_file, 'r') as f:
        data = f['features'][:]
        
        if hasattr(data.dtype, 'names') and data.dtype.names:
            print("✅ STRUCTURED format detected!")
            print(f"   Fields: {data.dtype.names}")
            print(f"   Dtype: {data.dtype}")
            
            # Check mask convention
            metadata = data['metadata'][0]
            n_valid = np.sum(metadata[:, 3] == 1.0)
            n_invalid = np.sum(metadata[:, 3] == 0.0)
            print(f"   Valid patches (mask=1.0): {n_valid}")
            print(f"   Invalid patches (mask=0.0): {n_invalid}")
            
        else:
            print("❌ FLAT format detected!")
            print(f"   Shape: {data.shape}")
            print(f"   Dtype: {data.dtype}")
            
            # Check mask convention
            metadata = data[0, :, :5]
            n_valid = np.sum(metadata[:, 3] == 0.0)
            n_invalid = np.sum(metadata[:, 3] == 1.0)
            print(f"   Valid patches (mask=0.0): {n_valid}")
            print(f"   Invalid patches (mask=1.0): {n_invalid}")

# Usage:
verify_extraction_format('dinov2_query_local.hdf5')
```

## 📋 Verification and Testing

### Format Detection Script
```python
def detect_and_analyze_format(hdf5_file):
    """
    Automatically detect format and provide detailed analysis
    """
    with h5py.File(hdf5_file, 'r') as f:
        data = f['features'][:]
        
        print(f"🔍 Analyzing: {hdf5_file}")
        print(f"📊 Data shape: {data.shape}")
        print(f"📊 Data dtype: {data.dtype}")
        
        if hasattr(data.dtype, 'names') and data.dtype.names:
            # Structured format
            print("✅ Format: STRUCTURED")
            print(f"   Fields: {list(data.dtype.names)}")
            
            metadata = data['metadata'][0]
            descriptors = data['descriptor'][0]
            
            print(f"   Metadata shape: {metadata.shape}, dtype: {metadata.dtype}")
            print(f"   Descriptors shape: {descriptors.shape}, dtype: {descriptors.dtype}")
            
            # Analyze masks
            valid_patches = np.sum(metadata[:, 3] == 1.0)
            invalid_patches = np.sum(metadata[:, 3] == 0.0)
            
            print(f"   Valid patches (mask=1.0): {valid_patches}")
            print(f"   Padding patches (mask=0.0): {invalid_patches}")
            
            # Sample valid patch
            valid_indices = np.where(metadata[:, 3] == 1.0)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[0]
                print(f"   Sample valid patch [{idx}]: x={metadata[idx,0]:.1f}, y={metadata[idx,1]:.1f}, scale={metadata[idx,2]:.3f}")
                
        else:
            # Flat format
            print("❌ Format: FLAT")
            
            if len(data.shape) == 3:
                metadata = data[0, :, :5]
                descriptors = data[0, :, 5:]
                
                print(f"   Metadata shape: {metadata.shape}, dtype: {metadata.dtype}")
                print(f"   Descriptors shape: {descriptors.shape}, dtype: {descriptors.dtype}")
                
                # Analyze masks
                valid_patches = np.sum(metadata[:, 3] == 0.0)
                invalid_patches = np.sum(metadata[:, 3] == 1.0)
                
                print(f"   Valid patches (mask=0.0): {valid_patches}")
                print(f"   Padding patches (mask=1.0): {invalid_patches}")
                
                # Sample valid patch
                valid_indices = np.where(metadata[:, 3] == 0.0)[0]
                if len(valid_indices) > 0:
                    idx = valid_indices[0]
                    print(f"   Sample valid patch [{idx}]: x={metadata[idx,0]:.1f}, y={metadata[idx,1]:.1f}, scale={metadata[idx,2]:.3f}")
        
        print("=" * 60)
```

## 🎯 Best Practices and Recommendations

### 1. Format Consistency
- **Always use structured format** for AMES evaluation
- **Convert existing flat format** data before evaluation
- **Verify format** after extraction using detection scripts

### 2. Mask Convention Compliance
```python
# Always follow structured format convention:
valid_patches = metadata[:, 3] == 1.0  # 1.0 = valid
invalid_patches = metadata[:, 3] == 0.0  # 0.0 = padding

# Never mix conventions within a dataset
```

### 3. Data Type Management
```python
# Use appropriate precision for memory efficiency:
metadata = metadata.astype(np.float32)      # Coordinates and scales
descriptors = descriptors.astype(np.float16) # Feature vectors (saves 50% memory)
```

### 4. Extraction Configuration
```python
# For AMES compatibility, always use:
desc_type = 'local,global'  # Triggers structured format
topk = 600                  # Standard patch count
```

## 🚨 Common Issues and Solutions

### Issue 1: "KeyError: 'descriptor'"
**Problem:** Trying to access structured format fields on flat format data.
**Solution:** Convert to structured format or use flat format access patterns.

### Issue 2: "No valid patches found"
**Problem:** Using wrong mask convention (0.0 vs 1.0).
**Solution:** Check and convert mask values during format conversion.

### Issue 3: "Shape mismatch in similarity calculation"
**Problem:** Mixing float16 and float32 precisions.
**Solution:** Cast to common precision before calculations.

### Issue 4: "Memory usage too high"
**Problem:** Using float32 for large descriptor arrays.
**Solution:** Use float16 for descriptors (compatible with original datasets).

## 🔗 Integration with AMES Pipeline

### Query Processing Workflow
```python
# 1. Extract query features in structured format
desc_type = 'local,global'
extract_descriptors.main()

# 2. Verify format compatibility
verify_extraction_format('dinov2_query_local.hdf5')

# 3. Load gallery features (assume structured)
gallery_data = load_structured_features('dinov2_gallery_local.hdf5')

# 4. Run AMES evaluation
results = ames_evaluate(query_data, gallery_data)
```

### Batch Processing
```python
# Convert multiple flat format files to structured
flat_files = [
    'dinov2_query_local.hdf5',
    'dinov2_gallery_local.hdf5',
    'cvnet_query_local.hdf5',
    'cvnet_gallery_local.hdf5'
]

for flat_file in flat_files:
    structured_file = flat_file.replace('.hdf5', '_structured.hdf5')
    convert_flat_to_structured(flat_file, structured_file)
    print(f"✅ Converted: {flat_file} -> {structured_file}")
```

## 📈 Performance Considerations

### Memory Usage
- **Structured format with float16**: ~50% less memory than flat float32
- **Recommended for large datasets**: Use structured format consistently

### Loading Speed
- **Structured format**: Slightly slower initial load, faster field access
- **Flat format**: Faster initial load, requires manual slicing

### Compatibility
- **Structured format**: Perfect compatibility with original Oxford/Paris datasets
- **Mixed formats**: Potential for subtle bugs and incorrect results

---

**💡 Remember:** When in doubt, use structured format for guaranteed compatibility with AMES evaluation and existing benchmark datasets.