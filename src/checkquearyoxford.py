import h5py
import numpy as np
import os

'''
dinov2_query_local.hdf5 → local descriptors for the queries
dinov2_gallery_local.hdf5 → local descriptors for the database/gallery
nn_dinov2.pkl → precomputed nearest neighbors
gnd_roxford5k.pkl → ground-truth info (which images are correct matches)
test_query.txt, test_gallery.txt → list of image IDs


1) Imports and Utility Functions

Imports h5py, numpy, and os for file and array operations.
Defines a normalize function for L2-normalizing vectors (important for cosine similarity).
Defines read_txt_ids to read image IDs from a text file (one per line).

2) File Paths

Sets the paths for the query and gallery HDF5 files.
Sets the paths for test_query.txt and test_gallery.txt, which contain the list of image IDs for queries and gallery images.

3) Load and Process Query

Opens the query HDF5 file and lists all keys (should be one per query).
Prints information about the query file structure and descriptor shapes.
Loads the first query’s descriptor, reshapes it to combine all local descriptors, and averages (pools) them to get a single global vector.
Normalizes the query vector.
Loads the corresponding query image filename from test_query.txt and prints it.

4) Load and Process Gallery

Reads gallery image IDs from test_gallery.txt.
Opens the gallery HDF5 file and checks if it contains a single dataset (e.g., 'features').
If so, extracts only the 'descriptor' field, reshapes and averages across all local descriptors for each image to get global vectors.
If not, falls back to per-image keys and processes each as above.
Normalizes all gallery vectors.

5) Compute Similarities and Ranking

Computes cosine similarity scores between the query vector and all gallery vectors.
Sorts the gallery images by similarity (descending).
Prints the top-k (default 5) most similar gallery images and their scores, using the IDs from test_gallery.txt.

6) Output

The script prints the query image filename, the query HDF5 key, and the top-k ranked gallery image filenames with their similarity scores.
This script is a minimal retrieval evaluation: it takes a query image, computes its global descriptor, compares it to all gallery images, and reports the most similar ones, using the correct mapping between HDF5 data and image filenames.



'''




# Normalize (important for cosine similarity)
def normalize(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def read_txt_ids(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# Load descriptors
query_file = "data/roxford5k/dinov2_query_local.hdf5"
gallery_file = "data/roxford5k/dinov2_gallery_local.hdf5"
query_txt = os.path.join(os.path.dirname(query_file), 'test_query.txt')
query_ids = read_txt_ids(query_txt)

# --- Load and process query ---
with h5py.File(query_file, "r") as fq:
    qids = list(fq.keys())
    if not qids:
        raise RuntimeError("No queries found in query file")
    qid = qids[0]
    entry = fq[qid][()]

    # --- Inspect query file ---
    query_keys = list(fq.keys())
    print(f"Query file contains {len(query_keys)} keys: {query_keys}")
    for k in query_keys:
        entry = fq[k][()]
        print(f"Key: {k}, entry shape: {entry.shape}")
        if 'descriptor' in entry.dtype.fields:
            desc = entry['descriptor']
            print(f"  descriptor shape: {desc.shape}, dtype: {desc.dtype}")

    # --- Load and process query ---
    qid = list(fq.keys())[0]
    entry = fq[qid][()]
    query_desc = entry['descriptor'].astype(np.float32)
    if query_desc.ndim == 3:
        query_desc = query_desc.reshape(-1, query_desc.shape[-1])  # (70*700, 768)
    query_vec = np.mean(query_desc, axis=0)  # (768,)
    query_vec = normalize(query_vec[None, :])  # (1, 768)
    print("Query vector shape:", query_vec.shape)

#gallery_vecs = np.vstack(gallery_vecs)  # (N, 768)
#gallery_vecs = normalize(gallery_vecs)

# --- Load and process gallery (robust to single-dataset format) ---




gallery_txt = os.path.join(os.path.dirname(gallery_file), 'test_gallery.txt')
gallery_ids = read_txt_ids(gallery_txt)

with h5py.File(gallery_file, "r") as fg:
    keys = list(fg.keys())
    print(f"Gallery file contains {len(keys)} keys: {keys}")
    # If only one key, assume it's a dataset of all features
    if len(keys) == 1:
        dset = fg[keys[0]]
        print(f"Single dataset: {keys[0]}, shape: {dset.shape}, dtype: {dset.dtype}")
        # dset shape: (N, 700, 768) or (N, M, 768)
        gallery_desc = dset['descriptor'].astype(np.float32)
        if gallery_desc.ndim == 3:
            gallery_desc = gallery_desc.reshape(gallery_desc.shape[0], -1, gallery_desc.shape[-1])
            gallery_vecs = np.mean(gallery_desc, axis=1)  # (N, 768)
        else:
            raise ValueError(f"Unexpected gallery dataset shape: {gallery_desc.shape}")
    else:
        # fallback: per-image keys
        gallery_vecs = []
        for gid in keys:
            entry = fg[gid][()]
            desc = entry['descriptor'].astype(np.float32)
            if desc.ndim == 3 and desc.shape[0] == 1:
                desc = desc.squeeze(0)
            pooled = np.mean(desc, axis=0)  # (768,)
            gallery_vecs.append(pooled)
        gallery_vecs = np.vstack(gallery_vecs)

gallery_vecs = normalize(gallery_vecs)
print("Gallery vectors shape:", gallery_vecs.shape)

# --- Compute similarities ---
scores = (gallery_vecs @ query_vec.T).squeeze()  # (N,)

# Rank gallery
if scores.size == 0:
    raise RuntimeError("No scores computed; check gallery vectors")

rank_idx = np.argsort(-scores)   # descending
top_k = min(5, len(gallery_ids))


# Print query text info if available
query_idx = 0  # since we use the first query in the file
if query_idx < len(query_ids):
    print(f"Query text: {query_ids[query_idx]}")
else:
    print(f"Query index {query_idx} not found in test_query.txt")

print("Query ID:", qid)
for i in range(top_k):
    idx = int(rank_idx[i])
    if idx >= len(gallery_ids):
        print(f"Warning: index {idx} out of range for gallery IDs of length {len(gallery_ids)}")
        continue
    gid = gallery_ids[idx]
    print(f"Top {i+1}: {gid}, Score={scores[idx]:.4f}")



'''
Query file contains 1 keys: ['features']
Key: features, entry shape: (70,)
  descriptor shape: (70, 700, 768), dtype: float16
Query vector shape: (1, 768)
Gallery file contains 1 keys: ['features']
Single dataset: features, shape: (4993,), dtype: [('metadata', '<f4', (700, 5)), ('descriptor', '<f2', (700, 768))]
Gallery vectors shape: (4993, 768)
Query text: jpg/all_souls_000013.jpg,0,768,1024
Query ID: features
Top 1: jpg/christ_church_000451.jpg,4,768,1024, Score=0.8908
Top 2: jpg/all_souls_000002.jpg,0,677,1024, Score=0.8904
Top 3: jpg/oxford_000623.jpg,12,683,1024, Score=0.8878
Top 4: jpg/all_souls_000140.jpg,0,768,1024, Score=0.8875
Top 5: jpg/all_souls_000209.jpg,0,768,1024, Score=0.8857

'''