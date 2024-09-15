from einops import einsum


def cosine_similarity(x, y, eps=1e-8):
    """
    Args:
        x: (L1, D)
        y: (L2, D)
    Return:
        sim: (L1, L2)

    """
    x_norm = x.norm(dim=-1, keepdim=True, p=2)
    y_norm = y.norm(dim=-1, keepdim=True, p=2).transpose(0, 1)
    sim = einsum(x, y, "l d, k d -> l k") / (x_norm * y_norm + eps)
    return sim