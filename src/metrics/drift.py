
import numpy as np
from scipy.stats import wasserstein_distance, entropy

def compute_wasserstein(map1, map2):
    """
    Compute Wasserstein distance between distributions of saliency values.
    """
    v1 = map1.flatten()
    v2 = map2.flatten()
    return wasserstein_distance(v1, v2)

def compute_spearman(map1, map2):
    """
    Compute Spearman rank correlation.
    """
    # Use scipy or simple pearson on ranks?
    # Simple pearson on ranks is efficient.
    from scipy.stats import spearmanr
    v1 = map1.flatten()
    v2 = map2.flatten()
    corr, _ = spearmanr(v1, v2)
    return corr
