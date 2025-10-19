# Global and Local Score Combination in AMES Reranking

This document explains how the AMES reranking function combines global and local similarity scores to improve image retrieval accuracy.

## Overview

The reranking process uses a two-stage approach:
1. **Global Retrieval**: Fast initial candidate selection using global descriptors
2. **Local Verification**: Detailed patch-level matching for precise reranking

## The Score Combination Formula

```python
s = 1. / (1. + torch.exp(-t * raw_sim))  # Step 1: Sigmoid activation
s = l * nn_sims[:, :k] + (1 - l) * s[:, :k]  # Step 2: Weighted combination
```

## Step-by-Step Breakdown

### Step 1: Sigmoid Activation on Local Scores
```python
s = 1. / (1. + torch.exp(-t * raw_sim))
```

**Variables:**
- **`raw_sim`**: Raw local similarity scores from detailed patch-level matching
- **`t` (temp)**: Temperature parameter that controls the "sharpness" of the sigmoid
- **Purpose**: Normalizes raw local scores to [0,1] range and applies temperature scaling

**Temperature Effects:**
- **High temp (t > 1)**: Makes similarities more "sharp" (closer to 0 or 1)
- **Low temp (t < 1)**: Makes similarities more "soft" (gradual transitions)
- **t = 1**: Standard sigmoid

### Step 2: Weighted Linear Combination
```python
s = l * nn_sims[:, :k] + (1 - l) * s[:, :k]
```

**Variables:**
- **`nn_sims`**: Global similarity scores (from initial nearest neighbor search)
- **`s`**: Local similarity scores (after sigmoid activation)
- **`l` (lambda)**: Weighting parameter between global and local scores
- **`k`**: Number of top candidates to rerank

**Weighting Scheme:**
- **λ = 0**: Pure local scoring `s = 0 * global + 1 * local`
- **λ = 1**: Pure global scoring `s = 1 * global + 0 * local`  
- **λ = 0.5**: Equal weight `s = 0.5 * global + 0.5 * local`

## What Each Score Type Captures

### Global Scores (`nn_sims`)
- **Fast initial retrieval** using global image descriptors
- **Broad similarity** based on overall image content
- **Efficient** but may miss fine-grained details
- **Good recall** - finds generally similar images

### Local Scores (`raw_sim → s`)
- **Detailed patch-level matching** using spatial verification
- **Precise geometric relationships** between image patches
- **Expensive** but more accurate for true matches
- **Good precision** - verifies actual object presence

## Example Scenario

```python
# Example with k=100, lambda=0.7, temp=2.0

# Global scores (from initial search)
nn_sims[:, :100] = [0.8, 0.75, 0.7, 0.65, ...]  # Fast global similarity

# Raw local scores (from patch matching)  
raw_sim[:, :100] = [2.5, 1.8, -0.5, 3.1, ...]   # Detailed local matching

# Step 1: Apply sigmoid with temperature
s = 1 / (1 + exp(-2.0 * [2.5, 1.8, -0.5, 3.1, ...]))
s = [0.924, 0.858, 0.377, 0.957, ...]

# Step 2: Combine with lambda=0.7
final_scores = 0.7 * [0.8, 0.75, 0.7, 0.65, ...] + 0.3 * [0.924, 0.858, 0.377, 0.957, ...]
final_scores = [0.837, 0.832, 0.603, 0.742, ...]
```

## Benefits of This Combination

1. **Best of Both Worlds**: Fast global retrieval + accurate local verification
2. **Robust Ranking**: Global scores provide stability, local scores add precision
3. **Tunable Balance**: Lambda parameter allows adaptation to different datasets
4. **Temperature Control**: Adjusts confidence in local matches

## Parameter Tuning

The code tests multiple combinations:
```python
for k, l, t in it.product(top_k, lamb, temp):
    # k: [100, 200, 500] - how many to rerank
    # l: [0.0, 1.0, 2.0] - global vs local weight  
    # t: [0.5, 1.0, 2.0] - temperature scaling
```

This finds the optimal balance between global efficiency and local accuracy for each specific dataset and query type.

## Code Flow in Rerank Function

### 1. Initial Setup
```python
nn = cache_nn.clone()  # Clone cached nearest neighbors
nn_sims = nn[0]        # Extract global similarity scores
nn_inds = nn[1].long() # Extract gallery image indices
```

### 2. Local Score Computation
```python
# For each query
for q_f, i in tqdm(query_loader):
    q_score = []
    # Sample only top-k candidates for detailed matching
    gallery_loader.batch_sampler.sampler = nn_inds[i, :max_k].T.tolist()
    
    # Compute local similarity with each candidate
    for db_f, j in tqdm(gallery_loader):
        current_scores = model(
            *list(map(lambda x: x.to(device, non_blocking=True), q_f)),
            *list(map(lambda x: x.to(device, non_blocking=True), db_f)))
        q_score.append(current_scores.cpu().data)
    scores.append(torch.stack(q_score).T)

raw_sim = torch.cat(scores)  # Concatenate all local scores
```

### 3. Score Combination and Reranking
```python
# Apply sigmoid and combine scores
s = 1. / (1. + torch.exp(-t * raw_sim))
s = l * nn_sims[:, :k] + (1 - l) * s[:, :k]

# Sort by combined scores
closest_dists, indices = torch.sort(s, dim=-1, descending=True)
closest_indices = torch.gather(nn_inds, -1, indices)

# Update rankings
ranks = deepcopy(nn_inds)
ranks[:, :k] = deepcopy(closest_indices)
```

### 4. Evaluation
```python
# Convert to numpy and evaluate
ranks = ranks.cpu().data.numpy().T
metrics, score, _ = compute_metrics(query_loader.dataset, ranks, gnd)
```

## Key Implementation Details

### Junk Image Handling
```python
if gnd is not None and 'junk' in gnd[0]:
    for i in range(len(cache_nn[0])):
        if hard:
            junk_ids = gnd[i]['junk'] + gnd[i]['easy']
        else:
            junk_ids = gnd[i]['junk']
        is_junk = np.in1d(cache_nn[1, i], junk_ids)
        # Move junk images to end of ranking
        nn[:, i] = torch.cat((cache_nn[:, i, ~is_junk], cache_nn[:, i, is_junk]), dim=1)
```

### Parameter Search
The function systematically evaluates all parameter combinations to find the optimal settings:
- **k values**: Different numbers of candidates to rerank
- **λ values**: Different global/local balance points
- **t values**: Different temperature scales

### Best Parameter Saving
```python
if save_scores:
    if 'val' in query_loader.dataset.name:
        k, l, t = max(out, key=out.get)  # Find best parameter combination
        with open('best_parameters', 'wt') as fid:
            fid.write(f'test_dataset.alpha=[{l}] test_dataset.temp=[{t}]')
```

## Mathematical Interpretation

The combination formula can be interpreted as:
- **Global component**: `l * nn_sims` provides broad similarity context
- **Local component**: `(1-l) * sigmoid(t * raw_sim)` adds geometric verification
- **Temperature scaling**: Controls how confident the local matches should be
- **Lambda weighting**: Balances computational efficiency vs accuracy

This hybrid approach leverages the strengths of both global (fast, good recall) and local (accurate, good precision) similarity measures for optimal image retrieval performance.