import torch
import matplotlib.pyplot as plt
import numpy as np

'''
load model 
create random features 2 images
compute similarity  global which is scalar
'''

# Load the pretrained model, and weights and set it to evaluation mode

model = torch.hub.load('pavelsuma/ames', 'dinov2_ames').eval()

# just view weights
# Optionally, inspect model weights if needed
#weights = model.state_dict()
#print(weights.keys())  # Shows top-level keys in the model's state_dict


# Simulate two sets of image features (100 patches, each with 768 dimensions)

img_1_feat = torch.randn(1, 100, 768)
img_2_feat = torch.randn(1, 100, 768)

# Compute similarity between the two sets of features
sim = model(src_local=img_1_feat, tgt_local=img_2_feat)

'''
The result sim is typically a similarity matrix of shape [1, 100, 100], 
where each entry represents the similarity between a patch from img_1_feat and a patch from img_2_feat.

'''
# global similarity score   scalar 1 
# Similarity shape: torch.Size([1])

print("Similarity shape:", sim.shape)
print("Similarity matrix:", sim)

print("Similarity value:", sim.item())

'''
# average similarity
print("Mean similarity:", sim.mean().item())
# max score
print("Max similarity:", sim.max().item())

simMatrix = model(src_local=img_1_feat, tgt_local=img_2_feat, return_matrix=True)
print("Similarity Matrix shape:", simMatrix.shape)


# Remove batch dimension and convert to numpy
sim_matrix = simMatrix.squeeze(0).detach().cpu().numpy()

plt.imshow(sim_matrix, cmap='hot', interpolation='nearest')
plt.title("Patch Similarity Heatmap")
plt.colorbar()
plt.show()
'''
