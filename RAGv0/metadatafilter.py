# pre metadatafiler
# for each metadata field, maintain a embedding mapping
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def retrieve(
    index,
    chunks,
    embeddings: np.array,
    embed_model: SentenceTransformer,
    query: str,
    top_k: int = 5,
    source: str = "Llama.md"
    
):
    source2ids = {}
    for i, chunk in enumerate(chunks):
        source2ids.setdefault(
            chunk.source, []
        ).append(i)
    query_vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    if source:
        candidate_ids = source2ids.get(source, [])
        if not candidate_ids:
            return []
        
        candidate_vectors = embeddings[candidate_ids]
        dim = candidate_vectors.shape[1]
        sub_index = faiss.IndexFlatIP(dim)
        sub_index.add(candidate_vectors)

        scores, sub_indices = sub_index.search(
            query_vec,
            top_k
        )
        results = []
        for rank, sub_idx in enumerate(sub_indices):
            original_idx = candidate_ids[sub_idx]
            results.append({
                "score": float(scores[0][rank]),
                "chunk": chunks[original_idx]
            })
        return results
    else:
        scores, indices = index.search(query_vec, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "score": float(scores[0][rank]),
                "chunk": chunks[idx]
            })

        return results


