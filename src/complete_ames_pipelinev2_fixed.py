#!/usr/bin/env python3
"""
Complete AMES Pipeline: From Raw Image to Search Results
Uses the exact same DINOv2 extraction method as the pre-computed gallery features
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import h5py
import pickle
import numpy as np
import os
import cv2
from pathlib import Path

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add the parent directory (ames root) to Python path to access extract folder
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from extract.spatial_attention_2d import SpatialAttention2d

def display_image(img, window_name="Image", wait_key=True):
    """
    Display an image using OpenCV
    Args:
        img: Image array (BGR format for cv2)
        window_name: Name of the display window
        wait_key: If True, wait for key press before continuing
    """
    cv2.imshow(window_name, img)
    if wait_key:
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()

def find_divisors(number):
    """Find divisors for reshaping features"""
    divisors = np.arange(1, int(np.sqrt(number)) + 1)
    divisors = divisors[number % divisors == 0]
    return divisors

def calculate_receptive_boxes(imsize, ps):
    """Calculate receptive field boxes for patches"""
    imsize = torch.tensor(imsize)
    pc_x, pc_y = imsize // ps
    
    loc = torch.arange(max(pc_x, pc_y)) * ps + (ps / 2)
    loc_x = loc[None, :pc_x]
    loc_y = loc[None, :pc_y]
    boxes = torch.stack([loc_x.repeat_interleave(pc_y, dim=1), loc_y.tile(pc_x)], dim=-1)
    boxes /= imsize
    return boxes

def get_local_features(local_features, local_weights, imsize, ps=14, topk=700):
    """Extract top-k local features exactly like in AMES extraction"""
    w = local_weights.flatten(start_dim=1)
    rf_boxes = calculate_receptive_boxes(imsize, ps)
    
    local_feature = local_features.flatten(start_dim=-2).permute(0, 2, 1)
    seq_len = min(local_feature.shape[1], topk)
    
    weights, ids = torch.topk(w, k=seq_len, dim=1)
    top_feats = torch.gather(local_feature, 1, ids[..., None].repeat(1, 1, local_feature.shape[-1]))
    locations = torch.gather(rf_boxes.cuda(), 1, ids[..., None].repeat(1, 1, 2))
    
    # Create the same format as in HDF5: [x, y, scale_enc, mask, weight, features...]
    local_info = torch.zeros((top_feats.shape[0], topk, 773))
    scale_enc = torch.zeros_like(weights)  # Single scale
    mask = torch.zeros_like(weights)  # No mask
    
    local_info[:, :seq_len] = torch.cat((
        locations,  # [x, y] 
        scale_enc[..., None],  # scale encoding
        mask[..., None],  # mask
        weights[..., None],  # attention weights
        top_feats  # 768D features
    ), dim=-1).cpu()
    
    local_info[:, seq_len:, 3] = 1  # Set mask for padded entries
    return local_info[0]  # Return single image

def load_dinov2_ames_style():
    """Load DINOv2 model exactly like in AMES extraction"""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    
    def dino_forward_hook(module, x, res):
        return res["x_norm_clstoken"], res["x_norm_patchtokens"], None
    
    model.forward = model.forward_features
    model.register_forward_hook(dino_forward_hook)
    return model

class CompletePipeline:
    def __init__(self, data_root="data", model_path="dinov2_ames.pt"):
        self.data_root = Path(data_root)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.dinov2_model = None
        self.ames_model = None
        self.detector = None
        
        # Data
        self.gallery_features = None
        self.gallery_names = None
        self.ground_truth = None
        
        print(f"Using device: {self.device}")
        
    def load_dinov2_and_detector(self):
        """Load DINOv2 model and spatial attention detector exactly like AMES"""
        print("Loading DINOv2 model and detector...")
        
        # Load DINOv2
        self.dinov2_model = load_dinov2_ames_style()
        self.dinov2_model.cuda()
        self.dinov2_model.eval()
        
        # Load spatial attention detector
        self.detector = SpatialAttention2d(768)  # 768 for DINOv2
        self.detector.cuda()
        self.detector.eval()
        
        # Load pretrained detector weights
        try:
            detector_url = 'http://ptak.felk.cvut.cz/personal/sumapave/public/ames/networks/dinov2_detector.pt'
            cpt = torch.hub.load_state_dict_from_url(detector_url)
            self.detector.load_state_dict(cpt['state'], strict=True)
            print("✅ Loaded pretrained spatial attention detector")
        except:
            print("⚠️ Could not load pretrained detector, using random weights")
        
        print("✅ DINOv2 and detector loaded successfully")
        return self.dinov2_model, self.detector
    
    def quantization_factor(self, side, scale, patch_size=14):
        """Calculate quantization factor to ensure dimensions are multiples of patch_size"""
        new_side = scale * side
        quantize_to = max(round(new_side / patch_size), 1.0)
        return scale / ((new_side / patch_size) / quantize_to)
    
    def extract_ames_style_features(self, image_path, topk=700):
        """Extract features using the exact AMES pipeline"""
        if self.dinov2_model is None:
            self.load_dinov2_and_detector()
        
        print(f"Processing image: {Path(image_path).name}")
        
        # Load and preprocess image exactly like AMES
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Original image shape: {img.shape}")
        
        # Apply patch size quantization like in extract scripts
        patch_size = 14  # DINOv2 patch size
        scale = 1.0  # Default scale
        
        # Calculate quantized scale factors for each dimension
        scale_x = self.quantization_factor(img.shape[1], scale, patch_size)
        scale_y = self.quantization_factor(img.shape[0], scale, patch_size)
        
        # Resize image to ensure dimensions are multiples of patch_size
        if scale_x != 1.0 or scale_y != 1.0:
            if scale_x < 1.0 or scale_y < 1.0:
                img = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
            else:
                img = cv2.resize(img, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            print(f"Resized image shape: {img.shape} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")
        
        # Verify dimensions are multiples of patch_size
        assert img.shape[0] % patch_size == 0, f"Image height {img.shape[0]} is not a multiple of patch size {patch_size}"
        assert img.shape[1] % patch_size == 0, f"Image width {img.shape[1]} is not a multiple of patch size {patch_size}"
        
        # Optional: Display the original image
        # cv2.imshow('Original Image', img)
        # cv2.waitKey(0)  # Wait for key press
        # cv2.destroyAllWindows()
        
        # Convert BGR to RGB and normalize like in extract scripts  
        img = img.astype(np.float32, copy=False)
        img = img[:, :, [2, 1, 0]]  # BGR to RGB
        img = img.transpose([2, 0, 1])  # HWC to CHW
        img = img / 255.0  # [0, 255] -> [0, 1]
        
        # Convert to tensor and normalize
        _MEAN = [0.485, 0.456, 0.406]
        _SD = [0.229, 0.224, 0.225]
        
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        # Normalize (CHW format)
        for c in range(3):
            img_tensor[0, c] = (img_tensor[0, c] - _MEAN[c]) / _SD[c]
        
        print(f"Final tensor shape: {img_tensor.shape}")
        
        with torch.no_grad():
            # Extract features using DINOv2
            cls, feats, weights = self.dinov2_model(img_tensor)
            
            # Reshape features like in AMES extraction
            div = find_divisors(feats.shape[1])
            feats = feats.permute(0, -1, -2).reshape(feats.shape[0], feats.shape[-1], div[-1], -1)
            
            # Apply spatial attention detector
            if self.detector:
                feats, weights = self.detector(feats)
            else:
                weights = torch.norm(feats, p=2, dim=1, keepdim=True)
            
            # Extract local features with same format as HDF5
            local_features = get_local_features(feats, weights, img_tensor.shape[-2:], ps=14, topk=topk)
            
            print(f"Local features shape: {local_features.shape}")
            
        return local_features.cpu()
    
    def load_ames_model(self):
        """Load trained AMES model using local class"""
        print(f"Loading AMES model...")
        
        try:
            # Import local AMES class
            from models.ames import AMES
            
            # Create model with same config as hub model
            self.ames_model = AMES(
                desc_name='dinov2', 
                local_dim=768,
                model_dim=128,
                nhead=2,
                num_encoder_layers=5,
                dim_feedforward=1024,
                binarized=True,  # Use non-binarized version
                pretrained='dinov2_ames.pt'  # This will download from hub
            )
            print("✅ Loaded local AMES model with hub weights")
                
        except Exception as e:
            print(f"Error loading local AMES model: {e}")
            # Fallback to hub model
            self.ames_model = torch.hub.load('pavelsuma/ames', 'dinov2_ames').eval()
            print("⚠️ Using hub model as fallback")
            
        self.ames_model.to(self.device)
        self.ames_model.eval()
        print("✅ AMES model loaded successfully")
        return self.ames_model
    
    def load_gallery_data(self, dataset='roxford5k'):
        """Load pre-computed gallery features and metadata"""
        gallery_path = self.data_root / dataset
        
        print(f"Loading gallery data from {gallery_path}...")
        
        # Load gallery features
        gallery_hdf5 = gallery_path / "dinov2_gallery_local.hdf5"
        
        if gallery_hdf5.exists():
            with h5py.File(gallery_hdf5, 'r') as f:
                print(f"HDF5 keys: {list(f.keys())}")
                
                # The HDF5 structure from your extraction should have 'features' dataset
                if 'features' in f:
                    data = f['features'][:]
                    print(f"Raw gallery data shape: {data.shape}")
                    print(f"Raw gallery data dtype: {data.dtype}")
                    
                    # Check if it's a structured array with 'descriptor' field
                    if data.dtype.names and 'descriptor' in data.dtype.names:
                        # Extract descriptor field: shape (num_images, 700, 768)
                        descriptors = data['descriptor']
                        print(f"Descriptors shape: {descriptors.shape}")
                        
                        # Convert from float16 to float32 and create tensor
                        self.gallery_features = torch.tensor(descriptors.astype(np.float32))
                        print(f"Gallery features shape: {self.gallery_features.shape}")
                    else:
                        # Fallback: assume it's a regular array and extract last 768 dimensions
                        self.gallery_features = torch.tensor(data[..., -768:].astype(np.float32))
                        print(f"Gallery features shape: {self.gallery_features.shape}")
                else:
                    raise ValueError(f"Expected 'features' key in HDF5 file, found: {list(f.keys())}")
                    
        else:
            raise FileNotFoundError(f"Gallery features not found: {gallery_hdf5}")
            
        # Load gallery image names
        gallery_txt = gallery_path / "test_gallery.txt"
        if gallery_txt.exists():
            with open(gallery_txt, 'r') as f:
                self.gallery_names = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.gallery_names)} gallery image names")
        
        # Load ground truth
        gnd_path = gallery_path / f"gnd_{dataset}.pkl"
        if gnd_path.exists():
            with open(gnd_path, 'rb') as f:
                self.ground_truth = pickle.load(f)
            print("✅ Ground truth loaded")
            
        return self.gallery_features, self.gallery_names
    
    def compute_similarities(self, query_features):
        """Compute similarities between query and all gallery images using AMES"""
        if self.ames_model is None:
            self.load_ames_model()
            
        if self.gallery_features is None:
            raise ValueError("Gallery features not loaded. Call load_gallery_data() first.")
            
        print("Computing similarities with AMES...")
        print(f"Query shape: {query_features.shape}")  # Should be [topk, 773]
        print(f"Gallery shape: {self.gallery_features.shape}")  # Should be [num_gallery, topk, 768]
        
        # Extract just the feature part from query (last 768 dims)
        query_feat = query_features[:, -768:].unsqueeze(0).to(self.device)  # [1, topk, 768]
        similarities = []
        
        # Process in batches
        batch_size = 50
        num_gallery = self.gallery_features.shape[0]
        
        with torch.no_grad():
            for i in range(0, num_gallery, batch_size):
                end_idx = min(i + batch_size, num_gallery)
                batch_gallery = self.gallery_features[i:end_idx].to(self.device)  # [batch, topk, 768]
                
                batch_sims = []
                for j in range(batch_gallery.shape[0]):
                    gallery_feat = batch_gallery[j:j+1]  # [1, topk, 768]
                    
                    # Compute similarity using AMES
                    sim = self.ames_model(src_local=query_feat, tgt_local=gallery_feat)
                    batch_sims.append(sim.cpu().item())
                
                similarities.extend(batch_sims)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {end_idx}/{num_gallery} gallery images")
        
        return torch.tensor(similarities)
    
    def get_top_matches(self, similarities, top_k=5):
        """Get top-k matches with scores and names"""
        sorted_indices = torch.argsort(similarities, descending=True)
        top_indices = sorted_indices[:top_k]
        top_scores = similarities[top_indices]
        
        results = []
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            result = {
                'rank': i + 1,
                'index': idx.item(),
                'score': score.item(),
                'image_name': self.gallery_names[idx.item()] if self.gallery_names else f"image_{idx.item()}"
            }
            results.append(result)
            
        return results
    
    def search_image(self, image_path, top_k=25):
        """Complete pipeline: image -> features -> AMES -> results"""
        print("="*60)
        print("AMES SEARCH PIPELINE")
        print("="*60)
        
        # Step 1: Extract features using AMES-style extraction
        query_features = self.extract_ames_style_features(image_path, topk=700)
        
        # Step 2: Load models and data (if not already loaded)
        if self.ames_model is None:
            self.load_ames_model()

        #load gallery data
        if self.gallery_features is None:
            self.load_gallery_data()
            
        # Step 3: Compute similarities
        similarities = self.compute_similarities(query_features)
        
        # Step 4: Get top matches
        top_matches = self.get_top_matches(similarities, top_k)
        
        # Step 5: Display results
        print("\n" + "="*60)
        print("SEARCH RESULTS")
        print("="*60)
        print(f"Query image: {Path(image_path).name}")
        print(f"Gallery size: {len(similarities)}")
        print(f"Global max similarity: {similarities.max().item():.4f}")
        print(f"Global min similarity: {similarities.min().item():.4f}")
        print(f"Global mean similarity: {similarities.mean().item():.4f}")
        
        print(f"\nTop {top_k} matches:")
        print("-" * 60)
        for result in top_matches:
            print(f"Rank {result['rank']:2d}: {result['image_name']:30s} | Score: {result['score']:.4f}")
        
        # Use matplotlib to display query image and top 25 results
     
        
        # Create figure with subplots: 1 row for query + 5 rows for 25 results (5x5 grid)
        fig = plt.figure(figsize=(15, 18))
        fig.suptitle('AMES Image Search Results - Top 25', fontsize=16, fontweight='bold')
        
        # Load and display query image (top center)
        query_img = cv2.imread(image_path)
        query_img_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Query image subplot (spans multiple columns for centering)
        ax_query = plt.subplot(6, 5, (1, 5))  # Top row, spans all 5 columns
        ax_query.imshow(query_img_rgb)
        ax_query.set_title(f'Query: {Path(image_path).name}', fontsize=12, fontweight='bold', pad=15)
        ax_query.axis('off')
        
        # Display top 25 results in 5x5 grid below query
        for i, result in enumerate(top_matches):
            if i >= 25:  # Only show top 25
                break
                
            # Extract base image name (remove any metadata after comma)
            base_name = result['image_name'].split(',')[0]
            
            # Construct full path to result image
            result_image_path = self.data_root / "roxford5k" / base_name
            
            print(f"Debug: Loading image from {result_image_path}")
            
            try:
                # Load result image
                result_img = cv2.imread(str(result_image_path))
                if result_img is not None:
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    # Calculate subplot position: 5 results per row, starting from row 2
                    row = 2 + (i // 5)  # Start from row 2 (row 1 is query)
                    col = (i % 5) + 1    # Columns 1-5
                    
                    # Bottom grid subplots
                    ax = plt.subplot(6, 5, (row - 1) * 5 + col)
                    ax.imshow(result_img_rgb)
                    ax.set_title(f'Rank {result["rank"]}\n{base_name}\nScore: {result["score"]:.4f}', 
                                fontsize=7, pad=3)
                    ax.axis('off')
                else:
                    print(f"Could not load result image: {result_image_path}")
                    
            except Exception as e:
                print(f"Error loading result image {result['image_name']}: {e}")
        
        plt.tight_layout()
        plt.show()

        return top_matches, similarities

def main():
    """Example usage"""
    # Configuration - Try the BEST query for testing (from diagnostic)
    image_path = r"C:\gitRepo\ames\data\roxford5k\jpg\radcliffe_camera_000519.jpg"
    data_root = r"C:\gitRepo\ames\data"   
    model_path = "dinov2_ames.pt"
    
    # Initialize pipeline
    pipeline = CompletePipeline(data_root=data_root, model_path=model_path)
    
    # Run search
    try:
        results, all_similarities = pipeline.search_image(image_path, top_k=25)
        
        # Additional analysis
        print(f"\nSimilarity statistics:")
        print(f"Max: {all_similarities.max().item():.4f}")
        print(f"Min: {all_similarities.min().item():.4f}")
        print(f"Mean: {all_similarities.mean().item():.4f}")
        print(f"Std: {all_similarities.std().item():.4f}")
        
        # Import and run diagnostic comparison
        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from query_diagnostic import analyze_query_ground_truth, compare_with_pipeline_results
            
            # Analyze ground truth
            query_name = Path(image_path).stem  # Get filename without extension
            gt_entry, gnd_data = analyze_query_ground_truth(data_root, query_name)
            
            # Compare results
            compare_with_pipeline_results(results, gt_entry, gnd_data)
            
        except Exception as e:
            print(f"Could not run diagnostic comparison: {e}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
