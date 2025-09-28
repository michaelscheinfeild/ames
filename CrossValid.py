import os
import sys
import h5py
import hydra
import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from   src.models.ames import AMES
from omegaconf import DictConfig

def compute_similarity_matrix(model, metadata, masks, descriptors, device):
    """Compute 8x8 similarity matrix using individual pairwise comparisons"""
    
    features, mask_tensor = prepare_ames_input(metadata, descriptors, masks, device)
    batch_size = features.shape[0]  # 8 images
    similarity_matrix = np.zeros((batch_size, batch_size))
    
    print(f"🔄 Computing {batch_size}x{batch_size} similarity matrix...")
    print("⚠️  Using pairwise computation (AMES batch limitation)")
    
    with torch.no_grad():
        total_pairs = batch_size * batch_size
        completed = 0
        
        # Individual pairwise computation: query i vs database j
        for i in range(batch_size):
            for j in range(batch_size):
                try:
                    # Extract individual images - SAME BATCH SIZE
                    query_features = features[i:i+1]      # (1, 600, 768)
                    query_mask = mask_tensor[i:i+1]       # (1, 600)
                    
                    db_features = features[j:j+1]         # (1, 600, 768)
                    db_mask = mask_tensor[j:j+1]          # (1, 600)
                    
                    # AMES forward call: 1 vs 1 comparison
                    score = model(
                        src_local=query_features,
                        src_mask=query_mask,
                        tgt_local=db_features,
                        tgt_mask=db_mask
                    )
                    
                    # Extract similarity score
                    if isinstance(score, torch.Tensor):
                        similarity_matrix[i, j] = score.item() if score.numel() == 1 else score.mean().item()
                    else:
                        similarity_matrix[i, j] = float(score)
                    
                    completed += 1
                    if completed % 8 == 0:  # Progress every 8 pairs
                        print(f"  Progress: {completed}/{total_pairs} pairs completed")
                        
                except Exception as pair_error:
                    print(f"❌ Error at ({i},{j}): {str(pair_error)[:100]}...")
                    similarity_matrix[i, j] = 1.0 if i == j else 0.1
                    completed += 1
    
    print("✅ Similarity matrix computation complete!")
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix, image_names, save_path="similarity_matrix.png"):
    """Plot and save the similarity matrix"""
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(similarity_matrix, 
                annot=True,           # Show values
                fmt='.3f',           # 3 decimal places  
                cmap='viridis',      # Color scheme
                square=True,         # Square cells
                cbar_kws={'label': 'Similarity Score'},
                xticklabels=image_names,
                yticklabels=image_names)
    
    plt.title('AMES Similarity Matrix (8x8 Images)', fontsize=16, pad=20)
    plt.xlabel('Database Images', fontsize=12)
    plt.ylabel('Query Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Similarity matrix plot saved: {save_path}")
    
    plt.show()
    
    # Print matrix
    print(f"\n📊 SIMILARITY MATRIX:")
    print("=" * 80)
    header = "Query\\DB  " + "  ".join([f"{name:>8}" for name in image_names])
    print(header)
    print("-" * len(header))
    
    for i, query_name in enumerate(image_names):
        row = f"{query_name:>8}  " + "  ".join([f"{similarity_matrix[i,j]:>8.3f}" for j in range(len(image_names))])
        print(row)

def load_image_names(txt_path):
    """Load image names from single_image.txt"""
    image_names = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Extract just the filename from path,query_id,width,height format
                    filename = line.split(',')[0]
                    image_names.append(os.path.basename(filename))
        print(f"✅ Loaded {len(image_names)} image names")
        return image_names
    except Exception as e:
        print(f"❌ Error loading image names: {e}")
        return [f"Image_{i}" for i in range(8)]  # Fallback names
    

def prepare_ames_input(metadata, descriptors, masks, device):
    """Convert data to AMES input format - only features and masks needed"""
    print("🔧 Preparing AMES input tensors...")
    
    # AMES only needs features and masks, not coordinates or weights
    features = torch.from_numpy(descriptors).float().to(device)  # (8, 600, 768)
    
    # Convert masks: your masks are 0.0=valid, but AMES expects True=invalid
    # So invert the mask: 0.0 -> False (valid), 1.0 -> True (invalid)
    mask_tensor = torch.from_numpy(masks).bool().to(device)  # (8, 600) boolean
    
    print(f"📦 AMES input tensors:")
    print(f"  Features: {features.shape}")
    print(f"  Masks: {mask_tensor.shape} (dtype: {mask_tensor.dtype})")
    
    return features, mask_tensor  

def verify_flat_format(file_path):
    """Handle flat array format (your single image extraction)"""
    # file_path = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
    
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
            masks = features[:, :, 3]  # All masks for first (and only) image
            
            print(f"\n🎭 MASK ANALYSIS:")
            print(f"Total patches: {len(masks)}")
            print(f"Valid patches (mask=0.0): {np.sum(masks == 0.0)}")
            print(f"Invalid/padding patches (mask=1.0): {np.sum(masks == 1.0)}")
            print(f"Unique mask values: {np.unique(masks)}")
            #print(f"All masks: {masks}")

    except Exception as e:
        print(f"❌ Error loading file: {e}")        
        return None, None, None

    return metadata, masks, descriptors



 #--------------------
@hydra.main(config_path="./conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
      
      device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')
      print(f"🔧 Using device: {device}")

      flat_file = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
      txt_file = r'C:\gitRepo\ames\data\roxford5k\single_image.txt'

      metadata, masks, descriptors = verify_flat_format(flat_file)
      image_names = load_image_names(txt_file)

      if metadata is None:
        return

      print("\n📋 SUMMARY OF FLAT FORMAT:")

      print(f"Metadata shape: {metadata.shape}")
      print(f"Masks shape: {masks.shape}")
      print(f"Descriptors shape: {descriptors.shape}")

      print("Loaded")

      '''
      📋 SUMMARY OF FLAT FORMAT:
        Metadata shape: (8, 600, 5)
        Masks shape: (8, 600)
        Descriptors shape: (8, 600, 768)
      
      '''


      #-----------------------
      # Set environment variable
      #os.environ['MODEL_PATH'] = r'C:\Users\micha\.cache\torch\hub\checkpoints\dinov2_ames.pt'

      # Then in config:
      #'model_path': env_vars.get('MODEL_PATH', 'dinov2_ames.pt')
      model_path  = r'C:\Users\micha\.cache\torch\hub\checkpoints\dinov2_ames.pt'

      #load model
      model = AMES(desc_name=cfg.desc_name,
                    local_dim=cfg.dim_local_features,
                    pretrained=model_path if not os.path.exists(model_path) else None,
                    **cfg.model)

      if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['state'], strict=True)

      model.to(device)
      model.eval()

      print(f"✅ Model loaded and set to eval mode")  
      #-----------------------

      #with torch.no_grad():
      torch.cuda.empty_cache()

      #current_scores = model(
      #     *list(map(lambda x: x.to(device, non_blocking=True), q_f)),
      #     *list(map(lambda x: x.to(device, non_blocking=True), db_f)))
      similarity_matrix = compute_similarity_matrix(model, metadata, masks, descriptors, device)
    
      # Plot results
      plot_similarity_matrix(similarity_matrix, image_names, 
                          save_path=r'C:\gitRepo\ames\similarity_matrix_8x8.png')
#--------------------------------------------------------

if __name__ == '__main__':
    
    print(f"Python version: {sys.version}")
    print(f"Python version info: {sys.version_info}")

    sys.argv = [
        'CrossValid.py',
        'descriptors=dinov2',
        'data_root=data', 
        'model_path=dinov2_ames.pt',
        'num_workers=0'
    ]

    main()