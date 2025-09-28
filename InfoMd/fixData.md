# AMES Data Format & Input Handling Summary

## 📊 **Two Data Formats Supported**

### **Format 1: Flat Array (float32/float16) - Your Data**
```python
# Shape: [batch_size, num_patches, 773]
# Structure: [x, y, scale, attention, mask, descriptor_768_dims...]
#             |---- 5 metadata -----|  |------ 768 features ------|

all_local.shape = [6, 400, 773]  # Your extracted data
local_feat = all_local[:, :400, 5:]    # [6, 400, 768] - features
metadata = all_local[:, :400, :5]      # [6, 400, 5] - [x,y,scale,attention,mask]
```

### **Format 2: Structured Array - Oxford/Paris**
```python
# HDF5 structured array with named fields
all_local['descriptor'].shape = [6, 700, 768]  # Feature vectors
all_local['metadata'].shape = [6, 700, 5]      # Position/attention data
```

## 🔧 **Variable Patch Input Handling**

### **Core Capability**
```python
# AMES can handle different patch counts
img_1_feat = torch.randn(22, 100, 768)  # 100 patches
img_2_feat = torch.randn(22, 130, 768)  # 130 patches
sim = model(src_local=img_1_feat, tgt_local=img_2_feat)
# Output: torch.Size([22]) - one score per pair
```

### **Cross-Attention Mechanism**
```python
# Creates attention matrix between all patch pairs
Q = img_1_feat  # [22, 100, 768]
K = img_2_feat  # [22, 130, 768]
attention_scores = Q @ K.transpose(-2, -1)  # [22, 100, 130]
# Every patch from img_1 attends to every patch from img_2
```

## 🚨 **Your Current Issues & Solutions**

### **Problem 1: Patch Count Mismatch**
```python
# ❌ Your data: 400 patches extracted
# ❌ Your config: trying to use 700 patches
# ✅ Solution: Use desc_num=400 or re-extract with topk=600
```

### **Problem 2: Wrong Data Loader**
```python
# ❌ Using custom TestDataset 
# ✅ Should use TensorFileDataset or fix your loader for flat arrays
```

### **Problem 3: Metadata Handling**
```python
# ✅ Correct mask extraction:
masks = torch.from_numpy(metadata[..., 3]).bool()
# metadata structure: [x, y, scale, mask, weight]
#                     [0, 1,   2,    3,    4  ]
```

## 🎯 **Recommended Fix for Your Pipeline**

### **1. Update Configuration**
```python
cfg = OmegaConf.create({
    'test_dataset': {
        'query_desc_num': 400,  # Match your extracted data
        'db_desc_num': 400,     # Match your extracted data
        # ...
    }
})
```

### **2. Fix TestDataset for Flat Arrays**
```python
def __getitem__(self, batch_index):
    # Handle flat array format correctly
    max_patches = min(self.desc_num, all_local.shape[1])  # Don't exceed available
    local_feat = all_local[:, :max_patches, 5:]     # Features
    metadata = all_local[:, :max_patches, :5]       # Metadata
    masks = torch.from_numpy(metadata[..., 3]).bool()  # Validity masks
    return (local_feat, masks), batch_index
```

### **3. Alternative: Re-extract with Correct Patch Count**
```bash
python extract/extract_descriptors.py --topk 600  # Extract 600 patches
# Then use desc_num=600 in config
```

## 🏛️ **Dataset Evaluation Protocols**

### **Oxford/Paris (Dual Evaluation)**
```python
if dataset.name.startswith(('roxford5k', 'rparis6k')):
    # Medium: excludes only junk
    # Hard: excludes junk + easy (only hard positives count)
    final_map = (medium_map + hard_map) / 2
```

### **Your Dataset Options**
```python
# Option 1: Use same protocol
self.name = 'roxford5k_500'  # Triggers dual evaluation

# Option 2: Single evaluation  
self.name = 'custom_dataset'  # Only medium evaluation
```

## 📊 **Key Metadata Fields**

| Index | Field | Description | Usage |
|-------|-------|-------------|--------|
| `[..., 0]` | x-coordinate | Patch horizontal position | Spatial info |
| `[..., 1]` | y-coordinate | Patch vertical position | Spatial info |
| `[..., 2]` | scale | Patch scale factor | Multi-scale |
| `[..., 3]` | **mask** | **Validity flag (1.0=valid, 0.0=padding)** | **Critical for attention** |
| `[..., 4]` | weight | Attention/importance weight | Optional |

## 🎯 **Action Plan**

1. **Fix patch count mismatch** (400 vs 600/700)
2. **Use correct data format handling** (flat array)
3. **Extract and use validity masks** from metadata[..., 3]
4. **Choose appropriate evaluation protocol** (single vs dual)
5. **Test with corrected configuration**

## 🔍 **BatchSampler vs DataLoader Usage**

### **Why BatchSampler Inside DataLoader**
```python
# Two-level batching for variable-length data
query_sampler = BatchSampler(SequentialSampler(query_set), batch_size=300, drop_last=False)
query_loader = DataLoader(query_set, sampler=query_sampler, batch_size=1, collate_fn=basic_collate)

# Level 1: BatchSampler creates batches of 300 indices
# Level 2: DataLoader uses batch_size=1 to take one batch at a time
# Result: Preserves list structure for variable-length tensors
```

### **Benefits for AMES**
- **No padding required** - each image keeps its natural patch count
- **Memory efficient** - no wasted computation on padding
- **Model compatibility** - AMES expects lists of variable-length tensors

## 🔧 **Ground Truth Data Structure**

### **Understanding metadata[..., 3]**
```python
# Example metadata for 2 images, 4 patches each:
metadata = [
    # Image 1: 3 valid patches, 1 padding
    [[100.5, 200.3, 1.2, 1.0, 0.8],  # Valid patch (mask=1.0)
     [150.2, 180.7, 1.1, 1.0, 0.9],  # Valid patch (mask=1.0)
     [200.1, 220.4, 0.9, 1.0, 0.7],  # Valid patch (mask=1.0)
     [0.0,   0.0,   0.0, 0.0, 0.0]],  # Padding (mask=0.0)
    
    # Image 2: 2 valid patches, 2 padding  
    [[80.3,  150.2, 1.0, 1.0, 0.85], # Valid patch
     [120.7, 190.8, 1.3, 1.0, 0.92], # Valid patch
     [0.0,   0.0,   0.0, 0.0, 0.0],  # Padding (mask=0.0)
     [0.0,   0.0,   0.0, 0.0, 0.0]]  # Padding (mask=0.0)
]

# Extract masks: metadata[..., 3]
masks = [[1.0, 1.0, 1.0, 0.0],  # Image 1: first 3 valid
         [1.0, 1.0, 0.0, 0.0]]  # Image 2: first 2 valid
```

### **Ground Truth Categories**
```python
# For each query image:
gt_entry = {
    'bbx': [[x1, y1, x2, y2], ...],           # Bounding boxes
    'easy': [145, 892, 1023, ...],            # Easy matches (good conditions)
    'hard': [2341, 3456, 4123, ...],          # Hard matches (difficult conditions)
    'junk': [567, 1234, 2890, ...]            # Junk images (ignore in evaluation)
}

# Evaluation uses:
# - Easy + Hard = relevant images
# - Junk = excluded from calculation
```

## 🎯 **Complete Solution Summary**

The key insight: **Your extracted data uses Format 1 (flat arrays) with 400 patches, but your pipeline was configured for 600+ patches, causing mismatches in data loading and processing.**

### **Quick Fix Checklist**
- [ ] Set `desc_num=400` to match extracted data
- [ ] Handle flat array format: `features = data[:, :, 5:]`
- [ ] Extract validity masks: `masks = metadata[..., 3]`
- [ ] Use proper BatchSampler + DataLoader pattern
- [ ] Choose evaluation protocol (single vs dual)
- [ ] Test with corrected configuration

### **Alternative: Re-extract Data**
```bash
# Re-extract with 600 patches to match AMES defaults
python extract/extract_descriptors.py \
    --config conf/descriptors/dinov2.yaml \
    --dataset_config conf/dataset/test_dataset.yaml \
    --gallery_list data/roxford5k/test_gallery_500.txt \
    --output_file data/roxford5k/dinov2_gallery_500_local.hdf5 \
    --topk 600
```

This comprehensive fix addresses all the data format mismatches and input handling issues in your AMES pipeline.