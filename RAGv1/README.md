### Phase B — Make it “real” (1–2 weeks)
- Goal: introduce the 6 production features most juniors miss:
#### Metadata filtering (doc_id, source, last_edited...)
- use Qdrant for pre metadatafilter
#### Chunk strategy (document-aware, not naive fixed size)
- Recursive chunking (indexing)
- Context Expansion (retrieval)
#### Hybrid retrieval (BM25 + vectors) when needed
Linear fusion
#### Reranking (cross-encoder style scoring)

#### Evaluation (RAGAS + golden set)

Observability & trace (log retrieval hits, scores, contexts, latency)

Deliverable: “RAG v1” with evaluation report and tracked regressions.
Metadata filter (Qdrant)
        ↓
Dense search (Qdrant)        for recall
BM25 search (rank_bm25)
        ↓
Fusion(top k)
        ↓
Cross-Encoder Rerank(top n)  for precision
        ↓
Retrieval Evaluation (golden context match: generation+retrieval evaluation)
        ↓
Neighbor expansion
        ↓
LLM
        ↓
Generation Evaluation (RAGAS)
        ↓
Observability & trace (simple log)