
# Two data formats supported:

if all_local.dtype in (np.float32, np.float16):
    local_feat = all_local[:, :self.desc_num, 5:]     # Features: last 768 dims
    metadata = all_local[:, :self.desc_num, :5]      # Metadata: first 5 dims
else:
    local_feat = all_local['descriptor'][:, :self.desc_num]  # Structured array
    metadata = all_local['metadata'][:, :self.desc_num]     # Structured array


## Format 1: Flat Array (float32/float16)
# Shape: [batch_size, num_patches, 773]
# Structure: [x, y, scale, attention, mask, descriptor_768_dims...]
#             |---- 5 metadata -----|  |------ 768 features ------|

all_local.shape = [6, 700, 773]  # 6 images, 700 patches, 773 total dims

local_feat = all_local[:, :600, 5:]    # [6, 600, 768] - descriptor features
metadata = all_local[:, :600, :5]      # [6, 600, 5] - [x,y,scale,attention,mask]


## Format 2: Structured Array
used in oxford 

# HDF5 structured array with named fields
all_local['descriptor'].shape = [6, 700, 768]  # Feature vectors
all_local['metadata'].shape = [6, 700, 5]      # Position/attention data


# metadata structure for each patch:
metadata[i, j, :] = [x, y, scale, mask, weight]
#                    [0, 1,   2,    3,    4  ]
masks = torch.from_numpy(metadata[..., 3]).bool()

# AMES Input Handling: Variable Patch Counts

## Overview

AMES (Attention-based Multi-scale Embedding Similarity) can handle images with different numbers of patches through its transformer-based architecture and cross-attention mechanisms.

## Input Example

```python
Nimg = 22
img_1_feat = torch.randn(Nimg, 100, 768)  # 22 images, 100 patches each
img_2_feat = torch.randn(Nimg, 130, 768)  # 22 images, 130 patches each

# AMES can compare these despite different patch counts (100 vs 130)
sim = model(src_local=img_1_feat, tgt_local=img_2_feat)
# Output: torch.Size([22]) - one similarity score per image pair
```

## How AMES Handles Variable Patch Counts

### 1. Transformer Architecture Foundation

AMES uses transformer-based attention mechanisms that are inherently designed for variable-length sequences:

- **No Fixed Sequence Length**: Unlike CNNs, transformers don't require fixed input dimensions
- **Dynamic Attention**: Each patch can attend to any patch in the other image
- **Flexible Processing**: Handles different numbers of patches seamlessly

### 2. Cross-Attention Mechanism

The core of AMES's flexibility lies in its cross-attention computation:

```python
# Inside AMES model:
Q = img_1_feat  # Query:  [22, 100, 768] 
K = img_2_feat  # Key:    [22, 130, 768]
V = img_2_feat  # Value:  [22, 130, 768]

# Cross-attention computation
attention_scores = Q @ K.transpose(-2, -1)  # [22, 100, 130]
attention_weights = softmax(attention_scores / sqrt(768))
attended_features = attention_weights @ V    # [22, 100, 768]
```

### 3. Attention Matrix Dimensions

The attention mechanism creates a full similarity matrix between all patch pairs:

```
Input Shapes:
- img_1_feat: [batch_size, seq_len_1, feature_dim] = [22, 100, 768]
- img_2_feat: [batch_size, seq_len_2, feature_dim] = [22, 130, 768]

Attention Matrix:
- Shape: [batch_size, seq_len_1, seq_len_2] = [22, 100, 130]
- Meaning: attention_matrix[i, j, k] = similarity between:
  * Patch j from image i in first set
  * Patch k from image i in second set
```

### 4. Aggregation to Single Similarity Score

AMES aggregates the patch-level similarities to produce one score per image pair:

```python
# Possible aggregation strategies:
# 1. Max Pooling: max(attention_matrix)
# 2. Mean Pooling: mean(attention_matrix) 
# 3. Weighted Sum: learned_weights * attention_matrix
# 4. Top-k Selection: top_k_patches(attention_matrix)

# Final output: [22] - one similarity score per image pair
```

## Mathematical Flow

### Step-by-Step Process

1. **Feature Encoding**: Both image sets are encoded to patch features
   ```
   img_1: [22, 100, 768] - 22 images, 100 patches, 768-dim features
   img_2: [22, 130, 768] - 22 images, 130 patches, 768-dim features
   ```

2. **Cross-Attention Computation**: 
   ```
   Attention = softmax(Q @ K^T / √d_k)
   Where Q from img_1, K,V from img_2
   Result: [22, 100, 130] attention matrix
   ```

3. **Feature Aggregation**:
   ```
   Attended_Features = Attention @ V
   Result: [22, 100, 768] weighted features
   ```

4. **Similarity Scoring**:
   ```
   Similarity = aggregate_function(Attended_Features)
   Result: [22] final similarity scores
   ```

## Key Advantages

### 1. Flexible Input Handling
- **Different Resolutions**: Images don't need same size
- **Variable Patch Counts**: Handles 100 vs 130 patches naturally
- **Aspect Ratio Independence**: Works with rectangular images
- **Crop Tolerance**: Partial images can be compared to full images

### 2. Learned Correspondences
- **No Manual Alignment**: Model learns best patch correspondences
- **Attention Weights**: Important patches get higher attention
- **Spatial Awareness**: Maintains spatial relationships through positional encoding
- **Multi-scale Matching**: Can match patches at different scales

### 3. Computational Efficiency
- **Parallel Processing**: All patch pairs computed simultaneously
- **GPU Optimization**: Matrix operations are highly optimized
- **Batch Processing**: Handles multiple image pairs efficiently

## Practical Applications

### Real-World Scenarios
1. **Image Retrieval**: Query image vs gallery images of different sizes
2. **Object Recognition**: Cropped objects vs full scene images
3. **Place Recognition**: Different viewpoints with varying visible areas
4. **Content-Based Search**: Partial matches and region-of-interest queries

### Use Cases in Your Code
```python
# Example scenarios that work with AMES:
query_patches = 50   # Mobile phone capture
gallery_patches = 200 # High-res database image

img_query = torch.randn(1, query_patches, 768)
img_gallery = torch.randn(1, gallery_patches, 768)

similarity = model(src_local=img_query, tgt_local=img_gallery)
# Returns: torch.Size([1]) - single similarity score
```

## Implementation Notes

### Current Code Structure
```python
# Your startModel.py implementation:
Nimg = 22
img_1_feat = torch.randn(Nimg, 100, 768)  # First set: 100 patches
img_2_feat = torch.randn(Nimg, 130, 768)  # Second set: 130 patches

sim = model(src_local=img_1_feat, tgt_local=img_2_feat)
# Output shape: torch.Size([22]) - batch of similarities

# Handling different output shapes:
if sim.numel() == 1:
    print("Similarity value:", sim.item())
else:
    print("Similarity values (batch):", sim)
    print("Mean similarity:", sim.mean().item())
    print("First similarity:", sim[0].item())
```

### Best Practices
1. **Consistent Feature Extraction**: Use same backbone (DINOv2) for both images
2. **Proper Normalization**: Ensure features are properly normalized
3. **Batch Processing**: Process multiple pairs efficiently
4. **Memory Management**: Large attention matrices can consume significant memory

## Technical Details

### Attention Mechanism
- **Multi-Head Attention**: Multiple attention heads capture different relationships
- **Scaled Dot-Product**: Prevents saturation in high-dimensional spaces
- **Residual Connections**: Enables deep network training
- **Layer Normalization**: Stabilizes training and inference

### Feature Dimensions
- **Input Features**: 768-dimensional (DINOv2 ViT-Base output)
- **Patch Count**: Variable (e.g., 100, 130, 200+ patches)
- **Batch Size**: Configurable (e.g., 1, 22, 64+ images)
- **Output**: Single similarity score per image pair

## Conclusion

AMES's transformer-based architecture makes it inherently capable of handling images with different numbers of patches. The cross-attention mechanism allows every patch from one image to attend to every patch in another image, regardless of the total patch count. This flexibility makes AMES particularly suitable for real-world image similarity tasks where images naturally vary in size, resolution, and content coverage.


## Fix Data for new dataset
we have seen two types of data Format 1: Flat Array (float32/float16) and Format 2: Structured Array when we use "C:\gitRepo\ames\extract\extract_descriptors.py" and THE CLASS "C:\gitRepo\ames\extract\image_dataset.py" what format the data saved for my dataset (not oxford or paris ) since i think i made mistake try to load 773 and didnt know how to handle this in input in ames and didnt use dthe good loaders used in "C:\gitRepo\ames\src\evaluate.py" as used batch loader also i didnt used the mask i tried to use 700 patches but i seen it use 600

Excellent observation! You've identified a critical mismatch between how you're creating data with extract_descriptors.py vs how the AMES evaluation expects to load it. Let me analyze this step by step.
🔍 What Format Does extract_descriptors.py Create?
Looking at [extract/image_dataset.py]image_dataset.py ), specifically the FeatureStorage class:

# class FeatureStorage:
## def __init__(self, save_dir, desc_name, split, extension,   global_desc_dim, local_desc_dim, dataset_size, save_type, topk=400):

        for desc_type in save_type:
            hdf5_file = h5py.File(os.path.join(save_dir, f'{desc_name}{split}_{desc_type}{extension}.hdf5'), 'w')
            shape = [dataset_size, topk, local_desc_dim + 5] if desc_type == 'local' else [dataset_size, global_desc_dim]
            #                    ^^^^  ^^^^^^^^^^^^^^^^^^
            #                    400   768 + 5 = 773
            hdf5_file.create_dataset("features", shape=shape, dtype=np.float32)


 ## Your Data Format is Format 1 (Flat Array):            
 # Your data shape from extract_descriptors.py:
shape = [dataset_size, 400, 773]  # [num_images, num_patches, features+metadata]
dtype = np.float32

# Structure per patch:
# [x, y, scale, attention, mask, descriptor_768_dims...]
# |---- 5 metadata -----|  |------ 768 features ------|
#  0  1    2       3       4     5...772

🚨 The Problems You Encountered:
1. Patch Count Mismatch


# extract_descriptors.py creates: 400 patches per image
topk = 400

# Your pipeline tries to use: 700 patches
# AMES evaluation expects: 600 patches (desc_num=600)

# Result: Index out of bounds or padding issues

2. Wrong Data Loader

# ❌ You used your custom TestDataset from tensor_dataset.py
# Which expects structured arrays or different format

# ✅ AMES evaluation uses TensorFileDataset  
# Which handles the flat array format properly

# Your data: [batch, 400, 773] where last 768 are features
# Your loader tries: local_feat = all_local[:, :700, 5:]  # Wrong! Only 400 patches available
# Correct should be: local_feat = all_local[:, :400, 5:]  # Use all available patches


# 🔧 Solutions:
Solution 1: Fix Your Data Extraction (Recommended)
Modify your feature extraction to match AMES expectations:


# When running extract_descriptors.py, use:
python extract/extract_descriptors.py \
    --config conf/descriptors/dinov2.yaml \
    --dataset_config conf/dataset/test_dataset.yaml \
    --gallery_list data/roxford5k/test_gallery_500.txt \
    --output_file data/roxford5k/dinov2_gallery_500_local.hdf5 \
    --topk 600  # ← Extract 600 patches instead of 400



# Solution 2: Use Correct Data Loader
The AMES evaluation uses TensorFileDataset, not your custom TestDataset. Looking at how it should work:    


# In get_test_sets() from dataset_loader.py:
query_set = TestDataset(
    desc_dir=test_dataset.desc_dir,
    desc_name=desc_name,
    desc_num=test_dataset.query_desc_num,  # 600
    gnd_data=test_gnd_data,
    nn_file=test_dataset.nn_file
)

# This TestDataset should handle flat array format:
# Shape: [num_images, 600, 773]
# Structure: [..., :5] = metadata, [..., 5:] = features


# Solution 3: Update Your tensor_dataset.py
Make your TestDataset compatible with the flat array format:


class TestDataset(Dataset):
    def __init__(self, desc_dir, desc_name, desc_num=None, gnd_data=None, nn_file=None):
        self.desc_dir = desc_dir
        self.desc_name = desc_name
        self.desc_num = desc_num or 600  # Default to 600
        self.gnd_data = gnd_data
        self.nn_file = nn_file
        
        # Load features - flat array format
        self.features_file = os.path.join(desc_dir, f"{desc_name}_gallery_local.hdf5")
        
        with h5py.File(self.features_file, 'r') as f:
            self.features = f['features'][:]  # [num_images, 400 or 600, 773]
            self.num_images = len(self.features)
            self.actual_patches = self.features.shape[1]  # 400 or 600
        
        # Adjust desc_num if necessary
        if self.desc_num > self.actual_patches:
            print(f"⚠️  Requested {self.desc_num} patches, but only {self.actual_patches} available")
            self.desc_num = self.actual_patches
        
        # Set dataset name
        self.name = 'roxford5k_500'
        
        # Load cache_nn
        self.cache_nn = self._load_cache_nn()
    
    def __getitem__(self, batch_index):
        """Handle batch loading for flat array format"""
        # Optimize HDF5 access
        idx = np.sort(np.unique(batch_index)).tolist()
        all_local = self.features[idx]  # [unique_images, patches, 773]
        
        # Restore original order
        all_local = all_local[[idx.index(i) for i in batch_index]]
        
        # Handle flat array format (Format 1)
        if all_local.dtype in (np.float32, np.float16):
            # Limit to available patches and requested desc_num
            max_patches = min(self.desc_num, all_local.shape[1])
            
            local_feat = all_local[:, :max_patches, 5:]     # [batch, patches, 768] - features
            metadata = all_local[:, :max_patches, :5]       # [batch, patches, 5] - metadata
        else:
            # Handle structured array format (Format 2) - if needed
            local_feat = all_local['descriptor'][:, :self.desc_num]
            metadata = all_local['metadata'][:, :self.desc_num]
        
        # Convert to tensors
        local_feat = torch.from_numpy(local_feat).float()
        masks = torch.from_numpy(metadata[..., 3]).bool()  # Extract validity masks
        
        return (local_feat, masks), batch_index


        📊 Correct Configuration for Your Data:

        # In your evaluate.py configuration:
cfg = OmegaConf.create({
    # ...
    'test_dataset': {
        'name': 'roxford',
        'desc_dir': r'C:\gitRepo\ames\data\roxford5k\Data500\roxford',
        'test_gnd_file': 'gnd_roxford5k_500.pkl',
        'nn_file': None,
        'query_desc_num': 400,    # ← Match your extracted data
        'db_desc_num': 400,       # ← Match your extracted data  
        'batch_size': 300,
        # ...
    }
})


🎯 Summary of Your Issues & Fixes:
Issue	Problem	Solution
Patch Count	Extract 400, try to use 700	Set desc_num=400 or re-extract with topk=600
Data Format	Flat array [400, 773]	Handle correctly in __getitem__
Loader Type	Wrong dataset class	Use correct TestDataset or fix yours
Masking	Not using masks	Extract metadata[..., 3] for masks
Batch Loading	Index errors	Proper batch index handling


🚀 Recommended Action Plan:

python extract/extract_descriptors.py --topk 600

Update your configuration:

'query_desc_num': 600,
'db_desc_num': 600,


Use the corrected TestDataset above

This should resolve all the format mismatches and get your evaluation working properly!