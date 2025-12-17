
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

def compute_ssim(map1, map2):
    """
    Compute SSIM between two attribution maps.
    Maps should be (H, W) or (1, H, W).
    Assumes maps are normalized [0, 1].
    """
    map1 = map1.squeeze()
    map2 = map2.squeeze()
    
    # data_range=1.0 because inputs are normalized
    return ssim(map1, map2, data_range=1.0)

def compute_cosine(map1, map2):
    """
    Compute cosine similarity between flattened maps.
    """
    v1 = map1.flatten().reshape(1, -1)
    v2 = map2.flatten().reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

def compute_topk_overlap(map1, map2, k_percent=0.1):
    """
    Compute Jaccard overlap of top k% pixels.
    """
    map1 = map1.flatten()
    map2 = map2.flatten()
    
    k = int(len(map1) * k_percent)
    
    # Get indices of top k elements
    # argsort returns typically ascending, so take last k
    top_k_1 = set(np.argsort(map1)[-k:])
    top_k_2 = set(np.argsort(map2)[-k:])
    
    intersection = len(top_k_1.intersection(top_k_2))
    union = len(top_k_1.union(top_k_2))
    
    return intersection / union if union > 0 else 0.0
