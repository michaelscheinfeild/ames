import os
import sys
import h5py
import hydra
import matplotlib
import numpy as np
import torch
import time  # Add this to your imports

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image

from   src.models.ames import AMES
from omegaconf import DictConfig


def create_individual_similarity_plots(similarity_matrix, image_names, save_path_folder, 
                                     image_folder_path, top_k=5):
    """
    Create individual plots for each image showing top-k most similar images
    
    Args:
        similarity_matrix: (N, N) numpy array of similarity scores
        image_names: List of image filenames
        save_path_folder: Output folder (e.g., 'C:\\github\\Results100')
        image_folder_path: Path to folder containing the actual images
        top_k: Number of top similar images to show (default 5)
    """
    
    # Create output directory
    os.makedirs(save_path_folder, exist_ok=True)
    print(f"📁 Creating similarity plots in: {save_path_folder}")
    
    num_images = len(image_names)
    
    for query_idx in range(num_images):
        query_name = image_names[query_idx]
        query_base_name = os.path.splitext(query_name)[0]  # Remove extension
        
        print(f"🖼️  Processing {query_idx+1}/{num_images}: {query_name}")
        
        # Get similarity scores for this query image
        similarities = similarity_matrix[query_idx]
        
        # Get top-k most similar images (excluding self)
        # Sort indices by similarity score in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter out self-similarity and get top-k
        top_indices = []
        for idx in sorted_indices:
            if idx != query_idx:  # Skip self
                top_indices.append(idx)
            if len(top_indices) >= top_k:
                break
        
        # Create the plot
        fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))
        fig.suptitle(f'Query: {query_name} - Top {top_k} Most Similar Images', 
                    fontsize=14, fontweight='bold')
        
        # Plot query image (leftmost)
        try:
            query_img_path = os.path.join(image_folder_path, query_name)
            if os.path.exists(query_img_path):
                img = Image.open(query_img_path)
                axes[0].imshow(img)
                axes[0].set_title(f'QUERY\n{query_name}\n(Self: {similarities[query_idx]:.3f})', 
                                fontsize=10, fontweight='bold', color='red')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, f'Query Image\nNot Found\n{query_name}', 
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('QUERY (Not Found)', fontsize=10, color='red')
                axes[0].axis('off')
        except Exception as e:
            axes[0].text(0.5, 0.5, f'Error Loading\n{query_name}', 
                       ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title(f'QUERY (Error)', fontsize=10, color='red')
            axes[0].axis('off')
        
        # Plot top-k similar images
        for i, similar_idx in enumerate(top_indices):
            similar_name = image_names[similar_idx]
            similar_score = similarities[similar_idx]
            
            try:
                similar_img_path = os.path.join(image_folder_path, similar_name)
                if os.path.exists(similar_img_path):
                    img = Image.open(similar_img_path)
                    axes[i + 1].imshow(img)
                    axes[i + 1].set_title(f'#{i+1}\n{similar_name}\nScore: {similar_score:.3f}', 
                                        fontsize=9)
                    axes[i + 1].axis('off')
                else:
                    axes[i + 1].text(0.5, 0.5, f'Image\nNot Found\n{similar_name}', 
                                   ha='center', va='center', transform=axes[i + 1].transAxes)
                    axes[i + 1].set_title(f'#{i+1} - Not Found\nScore: {similar_score:.3f}', 
                                        fontsize=9, color='orange')
                    axes[i + 1].axis('off')
            except Exception as e:
                axes[i + 1].text(0.5, 0.5, f'Error\n{similar_name}', 
                               ha='center', va='center', transform=axes[i + 1].transAxes)
                axes[i + 1].set_title(f'#{i+1} - Error\nScore: {similar_score:.3f}', 
                                    fontsize=9, color='red')
                axes[i + 1].axis('off')
        
        # Save the plot
        output_filename = f"{query_base_name}_similarity_top{top_k}.jpg"
        output_path = os.path.join(save_path_folder, output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close to free memory
        
        print(f"  ✅ Saved: {output_filename}")
    
    print(f"\n🎉 All {num_images} similarity plots created successfully!")
    print(f"📁 Output folder: {save_path_folder}")

def load_global_descriptors(global_file_path):
    """Load global descriptors from HDF5 file - they are already normalized"""
    try:
        with h5py.File(global_file_path, 'r') as f:
            global_features = f['features'][:]
            print(f"✅ Global descriptors loaded!")
            print(f"📊 Shape: {global_features.shape}")
            print(f"📋 Data type: {global_features.dtype}")
            
            # Global features are typically just the feature vectors (N, 768)
            if len(global_features.shape) == 2:
                return global_features
            elif len(global_features.shape) == 3 and global_features.shape[1] == 1:
                # If shape is (N, 1, 768), squeeze middle dimension
                return global_features.squeeze(1)
            else:
                print(f"⚠️  Unexpected global feature shape: {global_features.shape}")
                return global_features
                
    except Exception as e:
        print(f"❌ Error loading global descriptors: {e}")
        return None
    
def compute_similarity_matrix_withglobal(model, metadata, masks, descriptors, global_descriptors, 
                                       device, lamb=0.5, temp=1.0, save_path_folder=None):
    """
    Compute NxN similarity matrix combining global and local scores
    Global features are ALREADY NORMALIZED during extraction, so we use them directly
    
    Args:
        model: AMES model for local similarity computation
        metadata: Local feature metadata (N, 600, 5)
        masks: Local feature masks (N, 600) 
        descriptors: Local feature descriptors (N, 600, 768)
        global_descriptors: Global feature descriptors (N, 768) - ALREADY NORMALIZED
        device: Torch device
        lamb: Lambda weight between global (lamb) and local (1-lamb) scores
        temp: Temperature parameter for sigmoid activation on local scores
        save_path_folder: Output folder (e.g., 'C:\\github\\Results100Global')
        
    Returns:
        combined_matrix: (N, N) numpy array of combined similarity scores
        global_matrix: (N, N) numpy array of global similarity scores  
        local_matrix: (N, N) numpy array of local similarity scores (sigmoid)
    """
    
    # Start timing
    start_time = time.time()
    
    # Prepare local features for AMES
    features, mask_tensor = prepare_ames_input(metadata, descriptors, masks, device)
    batch_size = features.shape[0]  # N images
    
    # Global features are ALREADY NORMALIZED during extraction - use directly
    global_features = torch.from_numpy(global_descriptors).float().to(device)  # (N, 768)
    
    # Verify normalization (optional check)
    norms = torch.norm(global_features, dim=1)
    print(f"🔍 Global feature norms: min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f}")
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-2):
        print("✅ Confirmed: Global features are already normalized")
    else:
        print("⚠️  Warning: Global features may not be normalized, normalizing now...")
        global_features = torch.nn.functional.normalize(global_features, p=2, dim=1)
    
    # Initialize matrices
    local_similarity_matrix = np.zeros((batch_size, batch_size))
    global_similarity_matrix = np.zeros((batch_size, batch_size))
    
    print(f"🔄 Computing {batch_size}x{batch_size} similarity matrix with global+local combination...")
    print(f"⚖️  Lambda (global weight): {lamb}")
    print(f"🌡️  Temperature: {temp}")
    print(f"📊 Global descriptors shape: {global_features.shape}")
    
    with torch.no_grad():
        total_pairs = batch_size * batch_size
        completed = 0
        
        # Compute global similarity matrix - since features are normalized, cosine = dot product
        print("🌍 Computing global similarities (using normalized features)...")
        global_sim_tensor = torch.mm(global_features, global_features.t())  # Cosine similarity
        
        # Global similarities are already in [-1, 1], convert to [0, 1] to match local score range
        global_similarity_matrix = ((global_sim_tensor + 1.0) / 2.0).cpu().numpy()
        
        # Compute local similarity matrix using AMES
        print("🔍 Computing local similarities...")
        for i in range(batch_size):
            for j in range(batch_size):
                try:
                    # Extract individual images for local comparison
                    query_features = features[i:i+1]      # (1, 600, 768)
                    query_mask = mask_tensor[i:i+1]       # (1, 600)
                    
                    db_features = features[j:j+1]         # (1, 600, 768)
                    db_mask = mask_tensor[j:j+1]          # (1, 600)
                    
                    # AMES forward call: 1 vs 1 comparison
                    local_score = model(
                        src_local=query_features,
                        src_mask=query_mask,
                        tgt_local=db_features,
                        tgt_mask=db_mask
                    )
                    
                    # Extract local similarity score
                    if isinstance(local_score, torch.Tensor):
                        local_similarity_matrix[i, j] = local_score.item() if local_score.numel() == 1 else local_score.mean().item()
                    else:
                        local_similarity_matrix[i, j] = float(local_score)
                    
                    completed += 1
                    if completed % batch_size == 0:  # Progress every batch_size pairs
                        print(f"  Progress: {completed}/{total_pairs} pairs completed")
                        
                except Exception as pair_error:
                    print(f"❌ Error at ({i},{j}): {str(pair_error)[:100]}...")
                    local_similarity_matrix[i, j] = 1.0 if i == j else 0.0
                    completed += 1
    
    # Apply temperature scaling to local scores (sigmoid activation) - SAME AS RERANK
    print(f"🌡️  Applying temperature scaling (temp={temp})...")
    local_scores_sigmoid = 1.0 / (1.0 + np.exp(-temp * local_similarity_matrix))
    
    # Combine global and local scores using lambda weighting - SAME AS RERANK  
    print(f"⚖️  Combining scores (lambda={lamb})...")
    # EXACT SAME FORMULA: s = l * nn_sims[:, :k] + (1 - l) * s[:, :k]
    combined_similarity_matrix = lamb * global_similarity_matrix + (1 - lamb) * local_scores_sigmoid
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print statistics
    print("✅ Combined similarity matrix computation complete!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Global score range: [{np.min(global_similarity_matrix):.3f}, {np.max(global_similarity_matrix):.3f}]")
    print(f"📊 Local score range (raw): [{np.min(local_similarity_matrix):.3f}, {np.max(local_similarity_matrix):.3f}]")
    print(f"📊 Local score range (sigmoid): [{np.min(local_scores_sigmoid):.3f}, {np.max(local_scores_sigmoid):.3f}]")
    print(f"📊 Combined score range: [{np.min(combined_similarity_matrix):.3f}, {np.max(combined_similarity_matrix):.3f}]")
    
    # Save detailed results if path provided
    if save_path_folder:
        os.makedirs(save_path_folder, exist_ok=True)
        
        # Save individual matrices for analysis
        np.save(os.path.join(save_path_folder, 'global_similarity_matrix.npy'), global_similarity_matrix)
        np.save(os.path.join(save_path_folder, 'local_similarity_matrix_raw.npy'), local_similarity_matrix)
        np.save(os.path.join(save_path_folder, 'local_similarity_matrix_sigmoid.npy'), local_scores_sigmoid)
        np.save(os.path.join(save_path_folder, 'combined_similarity_matrix.npy'), combined_similarity_matrix)
        
        # Save parameters
        params_path = os.path.join(save_path_folder, 'combination_parameters.txt')
        with open(params_path, 'w') as f:
            f.write(f"Global + Local Similarity Combination Parameters\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Lambda (global weight): {lamb}\n")
            f.write(f"Temperature: {temp}\n")
            f.write(f"Matrix size: {batch_size}x{batch_size}\n")
            f.write(f"Computation time: {total_time:.2f} seconds\n")
            f.write(f"\nScore Statistics:\n")
            f.write(f"Global range: [{np.min(global_similarity_matrix):.3f}, {np.max(global_similarity_matrix):.3f}]\n")
            f.write(f"Local raw range: [{np.min(local_similarity_matrix):.3f}, {np.max(local_similarity_matrix):.3f}]\n")
            f.write(f"Local sigmoid range: [{np.min(local_scores_sigmoid):.3f}, {np.max(local_scores_sigmoid):.3f}]\n")
            f.write(f"Combined range: [{np.min(combined_similarity_matrix):.3f}, {np.max(combined_similarity_matrix):.3f}]\n")
        
        print(f"💾 Matrices and parameters saved to: {save_path_folder}")
    
    return combined_similarity_matrix, global_similarity_matrix, local_scores_sigmoid

def compute_similarity_matrix(model, metadata, masks, descriptors, device):
    """Compute NxN similarity matrix using individual pairwise comparisons"""
    
    # Start timing
    start_time = time.time()
    features, mask_tensor = prepare_ames_input(metadata, descriptors, masks, device)
    batch_size = features.shape[0]  # N images
    similarity_matrix = np.zeros((batch_size, batch_size))
    
    #print(f"🔄 Computing {batch_size}x{batch_size} similarity matrix...")
    #print("⚠️  Using pairwise computation (AMES batch limitation)")
    
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
                    if completed % batch_size == 0:  # Progress every batch_size pairs
                        print(f"  Progress: {completed}/{total_pairs} pairs completed")
                        
                except Exception as pair_error:
                    print(f"❌ Error at ({i},{j}): {str(pair_error)[:100]}...")
                    similarity_matrix[i, j] = 1.0 if i == j else 0.1
                    completed += 1
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    #print("✅ Similarity matrix computation complete!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Average time per pair: {total_time/total_pairs:.3f} seconds")
    print(f"🚀 Processing rate: {total_pairs/total_time:.1f} pairs/second")
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix, image_names, save_path="similarity_matrix.png"):
    """Plot similarity matrix optimized for 100 images"""
    
    num_images = len(image_names)
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Create heatmap without text annotations (too crowded for 100x100)
    sns.heatmap(similarity_matrix, 
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Similarity Score'},
                xticklabels=False,  # Don't show all 100 names
                yticklabels=False)  # Don't show all 100 names
    
    plt.title(f'AMES Similarity Matrix ({num_images}x{num_images} Images)', fontsize=16, pad=20)
    plt.xlabel('Database Images (0-99)', fontsize=12)
    plt.ylabel('Query Images (0-99)', fontsize=12)
    
    # Add some tick marks for reference
    tick_positions = range(0, num_images, 10)  # Every 10th image
    plt.xticks(tick_positions, [str(i) for i in tick_positions])
    plt.yticks(tick_positions, [str(i) for i in tick_positions])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Similarity matrix plot saved: {save_path}")
    plt.close()
    
    # Create separate mapping file
    mapping_path = save_path.replace('.png', '_image_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write("Image Index to Name Mapping:\n")
        f.write("=" * 60 + "\n")
        for i, name in enumerate(image_names):
            f.write(f"{i:3d}: {name}\n")
    
    print(f"✅ Image index mapping saved: {mapping_path}")

def plot_similarity_matrix_8_8(similarity_matrix, image_names, save_path="similarity_matrix.png"):
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

    
# Usage function to integrate with your existing code
def process_similarity_results(similarity_matrix, image_names, save_path_folder, 
                             image_folder_path, top_k=5):
    """
    Complete function to process similarity results and create all plots
    
    Args:
        similarity_matrix: (N, N) numpy array from compute_similarity_matrix()
        image_names: List of image names from load_image_names()
        save_path_folder: Output folder path (e.g., 'C:\\github\\Results100')
        image_folder_path: Path to folder containing actual image files
        top_k: Number of top similar images to show per query
    """
    
    print(f"\n🚀 PROCESSING SIMILARITY RESULTS")
    print(f"📊 Matrix size: {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]}")
    print(f"🖼️  Number of images: {len(image_names)}")
    print(f"📁 Output folder: {save_path_folder}")
    print(f"🖼️  Image source folder: {image_folder_path}")
    print(f"🔝 Top-K similar images per query: {top_k}")
    
    # Create individual similarity plots
    create_individual_similarity_plots(
        similarity_matrix=similarity_matrix,
        image_names=image_names,
        save_path_folder=save_path_folder,
        image_folder_path=image_folder_path,
        top_k=top_k
    )
    
    # Also create the overall similarity matrix plot
    overall_plot_path = os.path.join(save_path_folder, "overall_similarity_matrix.png")
    plot_similarity_matrix(similarity_matrix, image_names, save_path=overall_plot_path)
    
    print(f"\n✅ PROCESSING COMPLETE!")

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
 # to do use global_file too
 # todo use data loader batches 
@hydra.main(config_path="./conf", config_name="test", version_base=None)
def main(cfg: DictConfig):
      

      plt.ioff()
      matplotlib.use('Agg')

      device = torch.device('cuda:0' if torch.cuda.is_available() and not cfg.cpu else 'cpu')
      print(f"🔧 Using device: {device}")

      # oxford 100 data set extracted use run_extractSingle.py
      # seems use dino from hub and not same as configured dinov2_ames.pt 
      # 12 heads vs 2 heads also issu num encoder layer ?
      if 0:
        # Load global features
        global_file = r"C:\\github\\ames\\ames\\data\\roxford5k\\dinov2_query_global.hdf5"
        
        
        # Load flat format features and image names

        #flat_file = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
        #txt_file = r'C:\gitRepo\ames\data\roxford5k\single_image.txt'
        flat_file =r"C:\\github\\ames\\ames\\data\\roxford5k\\dinov2_query_local.hdf5"
        txt_file = r"C:\\github\\ames\\ames\\data\\roxford5k\\test_query_100.txt"

      # orhophoto data set
      # todo : add all gallery data set
      if 1:
            # Load global features
            global_file = r"C:\\OrthoPhoto\\data\\ortho\\dinov2_query_global.hdf5"
            
            
            # Load flat format features and image names

            #flat_file = r'C:\gitRepo\ames\data\roxford5k\dinov2_query_local.hdf5'
            #txt_file = r'C:\gitRepo\ames\data\roxford5k\single_image.txt'
            flat_file =r"C:\\OrthoPhoto\\data\\ortho\\dinov2_query_local.hdf5"
            txt_file = r"C:\\OrthoPhoto\\data\\ortho\\test_query.txt"

      global_descriptors = load_global_descriptors(global_file)
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
      # model_path  = r'C:\Users\micha\.cache\torch\hub\checkpoints\dinov2_ames.pt'
      model_path  = r'C:\Users\OPER\.cache\torch\hub\checkpoints\dinov2_ames.pt'    

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
      #plot_similarity_matrix(similarity_matrix, image_names, 
      #                    save_path=r'C:\gitRepo\ames\similarity_matrix_8x8.png')

      # Define paths
      if 0:
        results_folder = r'C:\github\Results100'
        image_source_folder = r'C:\github\ames\ames\data\roxford5k\jpg'  # Adjust to your image folder

      if 1:
        results_folder = r'C:\github\Results100Ortho'
        image_source_folder = r'C:\OrthoPhoto\Split'  # Adjust to your image folder

      # Process all similarity results
      process_similarity_results(
        similarity_matrix=similarity_matrix,
        image_names=image_names,
        save_path_folder=results_folder,
        image_folder_path=image_source_folder,
        top_k=5  )# Show top 5 similar images
      

      # global + local
      if global_descriptors is not None:
          # Use combined global+local computation
          lamb = 0.7  # 70% global, 30% local (experiment with this)
          temp = 1.0  # Standard temperature

          if 0:
            save_path_folder=r'C:\github\Results100Global'
                      
          if 1:
            save_path_folder=r'C:\github\Results100GlobalOrtho'
       

          combined_matrix, global_matrix, local_matrix = compute_similarity_matrix_withglobal(
              model=model,
              metadata=metadata,
              masks=masks,
              descriptors=descriptors,
              global_descriptors=global_descriptors,
              device=device,
              lamb=lamb,
              temp=temp,
              save_path_folder=save_path_folder
          )

          similarity_matrix = combined_matrix
          if 0:
           results_folder = r'C:\github\Results100Global'
           image_folder_path=r'C:\github\ames\ames\data\roxford5k\jpg'
          if 1:
           results_folder = r'C:\github\Results100GlobalOrtho'
           image_folder_path=r'C:\OrthoPhoto\Split'
      
          # Process results (same as before)
          process_similarity_results(
              similarity_matrix=similarity_matrix,
              image_names=image_names,
              save_path_folder=results_folder,
              image_folder_path=image_folder_path,
              top_k=5
          )


#--------------------------------------------------------
# run ams on 100 images and save images query results
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