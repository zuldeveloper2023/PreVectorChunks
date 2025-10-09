import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

def compute_reward(chunk_embeddings, min_len=5, max_len=50, sample_limit=200):
    """
    Compute reward for a chunk based on coherence and length.
    Penalize short chunks (<min_len) and reward coherence.
    """
    n = len(chunk_embeddings)
    if n == 0:
        return 0.0

    emb = chunk_embeddings.detach().cpu().numpy()
    if n > sample_limit:
        idx = np.random.choice(n, sample_limit, replace=False)
        emb = emb[idx]
        n = sample_limit

    sentence_reward = 0
    if n < min_len:
        sentence_reward = -5

    cos_sim = cosine_similarity(emb)
    if n > 1:
        avg_sim = np.mean(cos_sim[np.triu_indices_from(cos_sim, k=1)])
    else:
        avg_sim = 1.0

    return float(avg_sim + sentence_reward)
