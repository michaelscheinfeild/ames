#!/usr/bin/env python3
"""
Complete AMES Pipeline: From Raw Image to Search Results
Extracts DINOv2 features, loads AMES, and searches against gallery
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import h5py
import pickle
import numpy as np
import os
from pathlib import Path

# You'll need to import your AMES model class - adjust the import path
# from src.models.ames import AMES  # Adjust this import based on your code structure
'''
The CompletePipeline class is a high-level wrapper that automates the entire AMES image retrieval process, 
from raw image input to ranked search results. Here’s what each part does:

Initialization (__init__):
Sets up paths, device (CPU/GPU), and placeholders for models and data (DINOv2, AMES, gallery features, gallery names, ground truth).

load_dinov2:
Loads the DINOv2 model from torch.hub for feature extraction and sets up the required image preprocessing pipeline.

extract_dinov2_features:
Given an image path, loads and preprocesses the image, then extracts either local (patch) or global features using DINOv2.

load_ames_model:
Loads the AMES model, either from a local checkpoint or from torch.hub as a fallback, and moves it to the correct device.

load_gallery_data:
Loads precomputed gallery features from HDF5, gallery image names from a text file, and ground truth from a pickle file.

compute_similarities:
For a given query feature, computes similarity scores between the query and all gallery images using the AMES model,
 processing in batches for efficiency.

get_top_matches:
Sorts the similarity scores and returns the top-k matches, including their rank, index, score, and image name.

search_image:
The main pipeline: extracts features from a query image, loads models/data if needed, computes similarities, gets top matches, and prints results.

main (example usage):
Shows how to use the pipeline: initialize, run a search, and print similarity statistics.

In summary, CompletePipeline encapsulates all steps needed to go from a raw image to a ranked list of similar images in the gallery,
 using DINOv2 for feature extraction and AMES for similarity computation.
'''

def inspect_hdf5(filepath):
    """Inspect HDF5 file structure"""
    print(f"\n{'='*20} {filepath.name} {'='*20}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        for key in f.keys():
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset):
                print(f"\nDataset: {key}")
                print(f"  Shape: {dataset.shape}")
                print(f"  Dtype: {dataset.dtype}")
                print(f"  Size: {dataset.size * dataset.dtype.itemsize / (1024**3):.2f} GB")
                
                # Sample some values
                if len(dataset.shape) == 3:  # [num_images, patches, features]
                    print(f"  First image shape: {dataset[0].shape}")
                    print(f"  First patch, first 5 features: {dataset[0, 0, :5]}")
                elif len(dataset.shape) == 2:  # [num_images, features]
                    print(f"  First image, first 5 features: {dataset[0, :5]}")

class CompletePipeline:
    def __init__(self, data_root="data", model_path="dinov2_ames.pt"):
        self.data_root = Path(data_root)
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.dinov2_model = None
        self.ames_model = None
        
        # Data
        self.gallery_features = None
        self.gallery_names = None
        self.ground_truth = None
        
        print(f"Using device: {self.device}")
        
    def load_dinov2(self):
        """Load DINOv2 model for feature extraction"""
        print("Loading DINOv2 model...")
        
        # Load DINOv2 from torch hub
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').eval()
        self.dinov2_model.to(self.device)
        
        # Image preprocessing for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        print("✅ DINOv2 loaded successfully")
        return self.dinov2_model
    
    def extract_dinov2_features(self, image_path, return_local=True):
        """
        Extract DINOv2 features from an image
        Args:
            image_path: Path to image
            return_local: If True, return patch features; if False, return global features
        Returns:
            features: [num_patches, 768] if local, [768] if global
        """
        if self.dinov2_model is None:
            self.load_dinov2()
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        print(f"Processing image: {Path(image_path).name}")
        print(f"Image tensor shape: {input_tensor.shape}")
        
        with torch.no_grad():
            if return_local:
                # Get patch features (local features for AMES)
                features = self.dinov2_model.get_intermediate_layers(
                    input_tensor, n=1, return_class_token=False
                )[0]
                # Shape: [1, num_patches, 768] -> [num_patches, 768]
                features = features.squeeze(0)
                print(f"Local features shape: {features.shape}")
            else:
                # Get global features (CLS token)
                features = self.dinov2_model(input_tensor)
                # Shape: [1, 768] -> [768]
                features = features.squeeze(0)
                print(f"Global features shape: {features.shape}")
                
        return features.cpu()
    
    def load_ames_model(self):
        """Load trained AMES model"""
        print(f"Loading AMES model from {self.model_path}...")
        
        # You need to import your AMES class here
        # For now, using torch.hub as fallback
        try:
            # Try loading your local trained model
            if os.path.exists(self.model_path):
                # Load model architecture (you need to define this)
                # self.ames_model = AMES()
                # checkpoint = torch.load(self.model_path, map_location=self.device)
                # self.ames_model.load_state_dict(checkpoint)
                
                # Fallback to hub model for now
                self.ames_model = torch.hub.load('pavelsuma/ames', 'dinov2_ames').eval()
                print("⚠️ Using hub model instead of local weights")
            else:
                self.ames_model = torch.hub.load('pavelsuma/ames', 'dinov2_ames').eval()
                print("⚠️ Local model not found, using hub model")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback
            self.ames_model = torch.hub.load('pavelsuma/ames', 'dinov2_ames').eval()
            
        self.ames_model.to(self.device)
        print("✅ AMES model loaded successfully")
        return self.ames_model
    
    def load_gallery_data(self, dataset='roxford5k'):
        """Load pre-computed gallery features and metadata"""
        gallery_path = self.data_root / dataset
        
        print(f"Loading gallery data from {gallery_path}...")
        
        # Load gallery features
        gallery_hdf5 = gallery_path / "dinov2_gallery_local.hdf5"

        #inspect_hdf5(gallery_hdf5)

        if gallery_hdf5.exists():
            with h5py.File(gallery_hdf5, 'r') as f:
                print(f"HDF5 keys: {list(f.keys())}")
                if 'features' in f:
                    dset = f['features']
                elif len(f.keys()) == 1:
                    key = list(f.keys())[0]
                    dset = f[key]
                else:
                    key = list(f.keys())[0]
                    dset = f[key]

                # Extract only the 'descriptor' field: shape (num_images, 700, 768)
                desc = dset['descriptor'].astype(np.float32)  # (num_images, 700, 768)
                # Pool (mean) over patches to get (num_images, 768)
                desc = desc.mean(axis=1)
                self.gallery_features = torch.tensor(desc)
            print(f"Gallery features shape: {self.gallery_features.shape}")
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
        """
        Compute similarities between query and all gallery images
        Args:
            query_features: [num_patches, 768] query features
        Returns:
            similarities: [num_gallery] similarity scores
        """
        if self.ames_model is None:
            self.load_ames_model()
            
        if self.gallery_features is None:
            raise ValueError("Gallery features not loaded. Call load_gallery_data() first.")
            
        print("Computing similarities with AMES...")
        print(f"Query shape: {query_features.shape}")
        print(f"Gallery shape: {self.gallery_features.shape}")
        
        query_features = query_features.to(self.device)
        similarities = []
        
        # Process in batches to avoid memory issues
        batch_size = 100
        num_gallery = self.gallery_features.shape[0]
        
        with torch.no_grad():
            for i in range(0, num_gallery, batch_size):
                end_idx = min(i + batch_size, num_gallery)
                batch_gallery = self.gallery_features[i:end_idx].to(self.device)
                
                batch_sims = []
                for j in range(batch_gallery.shape[0]):
                    gallery_feat = batch_gallery[j]  # [num_patches, 768]
                    
                    # AMES expects [1, num_patches, 768] format
                    query_input = query_features.unsqueeze(0)  # [1, num_patches, 768]
                    gallery_input = gallery_feat.unsqueeze(0)  # [1, num_patches, 768]
                    
                    # Compute similarity
                    sim = self.ames_model(src_local=query_input, tgt_local=gallery_input)
                    batch_sims.append(sim.cpu().item())
                
                similarities.extend(batch_sims)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {end_idx}/{num_gallery} gallery images")
        
        return torch.tensor(similarities)
    
    def get_top_matches(self, similarities, top_k=5):
        """Get top-k matches with scores and names"""
        # Sort by similarity (descending)
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
    
    def search_image(self, image_path, top_k=5):
        """
        Complete pipeline: image -> features -> AMES -> results
        """
        print("="*60)
        print("AMES SEARCH PIPELINE")
        print("="*60)
        
        # Step 1: Extract DINOv2 features
        query_features = self.extract_dinov2_features(image_path, return_local=True)
        
        # Step 2: Load models and data (if not already loaded)
        if self.ames_model is None:
            self.load_ames_model()
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
        
        return top_matches, similarities

def main():
    """Example usage"""
    # Configuration
    image_path = r"C:\github\data\roxford5k\jpg\ashmolean_000063.jpg"
    #data_root = r"C:\github\data\\"
    data_root = r"C:\github\ames\ames\data\\"
    model_path = "dinov2_ames.pt"

    #draw image

    #img = Image.open(image_path)
    #img.show()
    
    # Initialize pipeline
    pipeline = CompletePipeline(data_root=data_root, model_path=model_path)
    
    # Run search
    try:
        results, all_similarities = pipeline.search_image(image_path, top_k=5)
        
        # Additional analysis
        print(f"\nSimilarity statistics:")
        print(f"Max: {all_similarities.max().item():.4f}")
        print(f"Min: {all_similarities.min().item():.4f}")
        print(f"Mean: {all_similarities.mean().item():.4f}")
        print(f"Std: {all_similarities.std().item():.4f}")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
