# AMES Data Format Analysis: Flat vs Structured Arrays

## 🔍 **Overview**

AMES feature extraction can produce two different HDF5 data formats depending on the `desc_type` parameter used during extraction. Understanding these formats is crucial for proper data loading and processing.

## 📊 **Format Comparison**

### **Format 1: Flat Array**
- **Shape**: `(N, patches, 773)`
- **Data Type**: `float32`
- **Structure**: Single homogeneous array
- **Access**: Direct indexing `features[i, j, k]`

### **Format 2: Structured Array** 
- **Shape**: `(N,)`
- **Data Type**: `[('metadata', '<f4', (patches, 5)), ('descriptor', '<f2', (patches, 768))]`
- **Structure**: Array of structs with named fields
- **Access**: Field-based `features['metadata']` and `features['descriptor']`

## 🔧 **What Controls the Format**

The format is determined by the `desc_type` parameter in the extraction command:

### **Flat Format (Format 1):**
```bash
# Single descriptor type produces flat format
python extract/extract_descriptors.py --desc_type "local"
```

### **Structured Format (Format 2):**
```bash
# Multiple descriptor types produce structured format  
python extract/extract_descriptors.py --desc_type "local,global"
```

## 📋 **Detailed Structure Analysis**

### **Format 1: Flat Array Structure**

```python
# File shape: (1, 600, 773)
# Data layout per patch (773 dimensions):

Column Index | Size | Content | Description
-------------|------|---------|------------
0-1          | 2    | x, y coordinates | Normalized patch positions [0,1]
2            | 1    | Scale encoding | Multi-scale level (usually 0)
3            | 1    | Mask | 0.0 = valid patch, 1.0 = padding
4            | 1    | Attention weight | Importance score (L2 norm or detector)
5-772        | 768  | Feature vector | DINOv2 descriptors

# Example access:
features = h5py.File('data.hdf5')['features'][:]
coordinates = features[0, :, :2]    # All patch coordinates
masks = features[0, :, 3]           # All mask values
descriptors = features[0, :, 5:]    # All feature vectors
```

### **Format 2: Structured Array Structure**

```python
# File shape: (70,) - array of structs
# Each element contains two named fields:

Field Name   | Shape      | Data Type | Content
-------------|------------|-----------|--------
'metadata'   | (700, 5)   | float32   | [x, y, scale, mask, weight]
'descriptor' | (700, 768) | float16   | DINOv2 feature vectors

# Example access:
features = h5py.File('data.hdf5')['features'][:]
metadata = features['metadata'][0]      # First image metadata (700, 5)
descriptors = features['descriptor'][0] # First image features (700, 768)
masks = metadata[:, 3]                  # Extract mask column
```

## 🎯 **Key Differences**

### **Data Types**
```python
# Flat Format:
# - Everything stored as float32
# - Larger file size but consistent precision

# Structured Format:  
# - Metadata: float32 (higher precision for coordinates)
# - Descriptors: float16 (lower precision, smaller files)
# - Better storage efficiency
```

### **Mask Interpretation**
```python
# Flat Format (Your single extraction):
# mask = 0.0 → Valid patch
# mask = 1.0 → Padding/invalid patch

# Structured Format (Original datasets):
# mask = 1.0 → Valid patch  
# mask = 0.0 → Padding/invalid patch
# Note: Inverted meaning!
```

### **Access Patterns**
```python
# Flat Format - Direct indexing:
patch_coords = features[image_idx, patch_idx, :2]
patch_mask = features[image_idx, patch_idx, 3]
patch_features = features[image_idx, patch_idx, 5:]

# Structured Format - Field-based access:
patch_coords = features['metadata'][image_idx, patch_idx, :2]
patch_mask = features['metadata'][image_idx, patch_idx, 3]  
patch_features = features['descriptor'][image_idx, patch_idx, :]
```

## 📊 **Real Examples from Your Data**

### **Your Single Image (Flat Format):**
```python
File: dinov2_query_local.hdf5
Shape: (1, 600, 773)
Data type: float32

# Sample patch data:
coordinates: [0.7585425, 0.24005753]  # x, y position
scale: 0.0                            # Single scale
mask: 0.0                             # Valid patch
weight: 3.45                          # Attention score
features: [0.123, -0.456, ...]       # 768 DINOv2 values

# All 600 patches are valid (mask=0.0)
```

### **Original Oxford Dataset (Structured Format):**
```python
File: dinov2_query_local.hdf5  
Shape: (70,)
Data type: [('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))]

# Access patterns:
metadata = features['metadata'][0]    # Shape: (700, 5)
descriptors = features['descriptor'][0] # Shape: (700, 768)

# Mask analysis shows mix of valid/padding:
# Image 0: 650 valid, 50 padding patches
# Image 1: 678 valid, 22 padding patches
```

## 🔧 **Conversion Between Formats**

### **Converting Flat to Structured:**
```python
def flat_to_structured(flat_features, topk=600):
    """Convert flat format to structured format"""
    n_images = flat_features.shape[0]
    
    # Create structured dtype
    dt = np.dtype([
        ('metadata', np.float32, (topk, 5)),
        ('descriptor', np.float16, (topk, 768))
    ])
    
    # Create structured array
    structured = np.empty(n_images, dtype=dt)
    
    for i in range(n_images):
        structured['metadata'][i] = flat_features[i, :, :5]
        structured['descriptor'][i] = flat_features[i, :, 5:].astype(np.float16)
    
    return structured
```

### **Converting Structured to Flat:**
```python
def structured_to_flat(structured_features):
    """Convert structured format to flat format"""
    n_images = len(structured_features)
    metadata = structured_features['metadata']
    descriptors = structured_features['descriptor']
    
    topk = metadata.shape[1]
    flat = np.empty((n_images, topk, 773), dtype=np.float32)
    
    for i in range(n_images):
        flat[i, :, :5] = metadata[i]
        flat[i, :, 5:] = descriptors[i].astype(np.float32)
    
    return flat
```

## 🎯 **Which Format to Use?**

### **Use Flat Format When:**
- Single descriptor type extraction
- Simple data processing
- Consistent data types needed
- Working with your own datasets

### **Use Structured Format When:**
- Multiple descriptor types (local + global)
- Storage efficiency important
- Compatibility with original Oxford/Paris datasets
- Following standard AMES evaluation protocols

## 🔄 **How to Generate Structured Format**

To create structured format like the original datasets:

```python
# Modify your extraction script:
desc_type = 'local,global'  # Instead of just 'local'

# This creates two files:
# - dinov2_query_local.hdf5  (structured format)
# - dinov2_query_global.hdf5 (global features)
```

## 📋 **Verification Script**

```python
def verify_data_format(file_path):
    """Automatically detect and verify data format"""
    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        
        if features.dtype.names:
            print("🔧 STRUCTURED FORMAT")
            print(f"Fields: {features.dtype.names}")
            print(f"Shape: {features.shape}")
            
            # Access structured data
            metadata = features['metadata'][0]
            masks = metadata[:, 3]
            valid_patches = np.sum(masks == 1.0)  # Note: 1.0 = valid
            
        else:
            print("🔧 FLAT FORMAT") 
            print(f"Shape: {features.shape}")
            print(f"Data type: {features.dtype}")
            
            # Access flat data
            masks = features[0, :, 3]
            valid_patches = np.sum(masks == 0.0)  # Note: 0.0 = valid
        
        print(f"Valid patches: {valid_patches}")
        return features
```

## 🎯 **Summary**

The choice between flat and structured formats affects:
- **Storage efficiency** (structured uses float16 for features)
- **Access patterns** (direct indexing vs field-based)  
- **Compatibility** (structured matches original datasets)
- **Mask interpretation** (inverted meanings!)

For maximum compatibility with AMES evaluation and original datasets, use structured format by extracting with `desc_type='local,global'`.