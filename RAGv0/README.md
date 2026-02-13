pip install sentence-transformers
conda install -c pytorch faiss-cpu

Phase A — Understand the pipeline (2–3 days)

Goal: build the simplest working RAG, then improve it step by step.

They must understand:

indexing: load → clean → chunk → embed → store

query: rewrite → retrieve → (hybrid optional) → rerank → synthesize → cite

failure modes: wrong chunking, stale docs, hallucinations, missing citations, “I don’t know” behavior

Deliverable: “RAG v0” that answers questions from a small doc set.