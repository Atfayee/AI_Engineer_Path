import json
from pathlib import Path
from main import build_pipeline, retrieval


# ==========================
# Load Golden Set
# ==========================
def load_golden_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# ==========================
# Recall@K
# ==========================
def compute_recall_at_k(pipeline, golden_set, k=5):
    hits = 0
    for sample in golden_set:
        question = sample["question"]
        golden_chunks = {
            (g["doc_id"], g["chunk_id"])
            for g in sample["golden_chunks"]
        }
        retrieved_chunks = retrieval(pipeline=pipeline, query=question, top_k=k)
        retriebed_ids = {
            (c.doc_id, c.chunk_id)
            for c in retrieved_chunks
        }
        hit = len(golden_chunks & retriebed_ids) > 0
        if hit:
            hits += 1
    recall = hits / len(golden_set)
    return recall

# ==========================
# MRR (Mean Reciprocal Rank)
# ==========================

def compute_mrr(pipeline, golden_set, k=5):
    reciprocal_ranks = []
    for sample in golden_set:
        question = sample["question"]
        golden_chunks = {
            (g["doc_id"], g["chunk_id"])
            for g in sample["golden_chunks"]
        }
        retrieved_chunks = retrieval(pipeline=pipeline, query=question, top_k=k)
        rank = None
        for idx, c in enumerate(retrieved_chunks):
            if (c.doc_id, c.chunk_id) in golden_chunks:
                rank = idx + 1
                break
        if rank:
            reciprocal_ranks.append(1/rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks / len(golden_set))

# ==========================
# Main Evaluation
# ==========================
def main():
    golden_path = Path(__file__).parent / "golden_set.json"
    golden_set = load_golden_set(golden_path)
    pipeline = build_pipeline()
    recall_at_5 = compute_recall_at_k(pipeline, golden_set, k=5)
    mrr_at_5 = compute_mrr(pipeline, golden_set, k=5)

    print("\n=== Evaluation Results ===\n")
    print(f"Recall@5: {recall_at_5:.3f}")
    print(f"MRR@5: {mrr_at_5:.3f}")


if __name__ == "__main__":
    main()