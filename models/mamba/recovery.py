import torch
from scipy.spatial import cKDTree

def align_features_batched(original_points, changed_points, changed_features):
    B, N, C = changed_features.size()
    aligned_features = torch.zeros((B, N, C), dtype=changed_features.dtype, device=changed_features.device)

    for b in range(B):
        
        original_points_np = original_points[b].cpu().numpy()
        changed_points_np = changed_points[b].cpu().numpy()
        changed_features_np = changed_features[b].cpu().numpy()

        
        tree = cKDTree(changed_points_np)
        distances, indices = tree.query(original_points_np, k=1)

        
        valid = distances < 1e-5

        
        aligned_features_b = torch.zeros_like(changed_features[b], dtype=changed_features.dtype, device=changed_features.device)
        aligned_features_b[valid] = torch.from_numpy(changed_features_np[indices[valid]]).to(aligned_features.device)
        
        aligned_features[b] = aligned_features_b

    return aligned_features
