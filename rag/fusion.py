"""Reciprocal Rank Fusion: combine lexical and dense rankings without tuning weights."""


def rrf(rankings: list, k: int = 60, top: int = 5, weights=None) -> list:
    """rankings: list of lists of docs (best first). weights: per-ranking multiplier."""
    weights = weights or [1.0] * len(rankings)
    scores, pool = {}, {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            key = doc["id"]
            pool[key] = doc
            scores[key] = scores.get(key, 0.0) + w * (1.0 / (k + rank + 1))
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"score": round(s, 4), **pool[did]} for did, s in ordered[:top]]
