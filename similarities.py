import numpy as np

def compute_cosine_similarity(w1, w2):
    """
    Compute the cosine similarity between two weight vectors.

    Args:
    w1, w2: NumPy arrays representing the weight vectors.

    Returns:
    cosine_similarity: A scalar representing the cosine similarity.
    """
    # Compute the dot product of the vectors
    dot_product = np.dot(w1, w2)

    # Compute the norms of the vectors
    norm_w1 = np.linalg.norm(w1)
    norm_w2 = np.linalg.norm(w2)

    # Compute the cosine similarity
    # Avoid division by zero
    if norm_w1 == 0 or norm_w2 == 0:
        cosine_similarity = 0  # Could also choose to return None or raise an exception
    else:
        cosine_similarity = dot_product / (norm_w1 * norm_w2)

    return cosine_similarity

def compute_weighted_cosine_similarity(w1, w2, weights):
    """
    Compute the weighted cosine similarity between two weight vectors.

    Args:
    w1, w2: NumPy arrays representing the weight vectors.
    weights: NumPy array representing the importance of each dimension.

    Returns:
    cosine_similarity: A scalar representing the weighted cosine similarity.
    """
    # Compute the dot product of the weighted vectors
    dot_product = np.dot(w1 * weights, w2 * weights)

    # Compute the norms of the weighted vectors
    norm_w1 = np.linalg.norm(w1 * weights)
    norm_w2 = np.linalg.norm(w2 * weights)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (norm_w1 * norm_w2)

    return cosine_similarity


def compute_weight_overlap(w1, w2, threshold=0):
    """
    Compute the weight overlap between two weight vectors.

    Args:
    w1, w2: NumPy arrays representing the weight vectors.
    threshold: A value below which weights are considered zero.

    Returns:
    overlap: A scalar representing the weight overlap.
    """
    # Create binary masks for the two weight vectors
    m1 = (np.abs(w1) > threshold).astype(int)
    m2 = (np.abs(w2) > threshold).astype(int)

    # Compute the overlap
    overlap = np.sum(m1 * m2)

    return overlap


def jaccard_similarity(w1, w2):
    a = sum(w1)
    overlap = np.sum(w1 * w2)
    common = round(overlap / a, 3)
    return common
    # intersection = (w1 * w2).sum()
    # union = w1.sum() + w2.sum() - intersection
    # if union == 0:
    #     return 0
    # else:
    #     return intersection / union