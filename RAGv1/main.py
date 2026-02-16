from dotenv import load_dotenv
import os, re, uuid, sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models
from trace_logger import TraceLogger

load_dotenv()


# ==========================
# Config
# ==========================

COLLECTION = "rag_v1"
TOP_K = 5
ALPHA = 0.6
SOURCE = None


# ==========================
# LLM
# ==========================
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ==========================
# Models
# ==========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

logger = TraceLogger()
# ==========================
# Data Structures
# ==========================

@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    source: Path
    text: str
    last_edited: float

class RAGPipeline:
    def __init__(self, chunks: list[Chunk], doc_chunk_map: dict, bm25: BM25Okapi, client_q: QdrantClient):
        self.chunks = chunks
        self.doc_chunk_map = doc_chunk_map
        self.bm25 = bm25
        self.client_q = client_q

# ==========================
# Utilities
# ==========================

def load_docs(data_dir: Path) -> list[dict]:
    docs = []
    for p in Path(data_dir).glob("**/*.md"):
        docs.append({
            "doc_id": p.name,
            "text": p.read_text(encoding="utf-8", errors="ignore"),
            "source": p,
            "last_edited": p.stat().st_mtime
        })
    return docs

def clean_text(text: str) -> str:
    t = text.replace("\r\n", "\n")
    t = "\n".join(l for l in t.splitlines())
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t


# ==========================
# Chunking
# ==========================

def split_by_headers(text: str):
    sections = re.split(r"(?=^#{1,6} )", text, flags=re.MULTILINE)
    return [s.strip() for s in sections]

def recursive_chunks(section: str, max_size=1000) -> list[str]:
    if len(section) < max_size:
        return [section]
    
    chunks = []
    paragraphs = section.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) < max_size:
            current += p + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            if len(p) > max_size:
                for i in range(0, len(p), max_size):
                    chunks.append(p[i:i+max_size])
                current = ""
            else:
                current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

def chunk_text(doc: dict) -> list[Chunk]:
    sections = split_by_headers(doc["text"])
    chunks= []
    cid = 0
    for sec in sections:
        sub_chunks = recursive_chunks(sec)
        for sc in sub_chunks:
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc["doc_id"],
                source=doc["source"],
                text=sc,
                last_edited=doc["last_edited"]
            ))
            cid += 1
    return chunks



# ==========================
# Build Pipeline
# ==========================

def build_pipeline():
    base_path = Path(__file__).parent
    data_dir = base_path / "data"
    docs = load_docs(data_dir=data_dir)

    chunks: list[Chunk] = []
    for d in docs:
        d["text"] = clean_text(d["text"])
        chunks.extend(chunk_text(d))
    
    doc_chunk_map = defaultdict(list)
    for c in chunks:
        doc_chunk_map[c.doc_id].append(c)
    for k in doc_chunk_map:
        doc_chunk_map[k].sort(key=lambda x:x.chunk_id)
    texts = [c.text for c in chunks]

    # BM25
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Dense embeddings
    embeddings = embed_model.encode(texts, normalize_embeddings=True).astype("float32")
    dim = embeddings.shape[1]

    # Qdrant
    client_q = QdrantClient(url="http://localhost:6333")
    client_q.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE
        )
    )
    points = []
    for c, v in zip(chunks, embeddings):
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=v,
            payload={
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source
            }
        ))
    client_q.upsert(collection_name=COLLECTION, points=points)
    return RAGPipeline(chunks, doc_chunk_map, bm25, client_q)



# ==========================
# Retrieval
# ==========================

def retrieval(pipeline: RAGPipeline, query: str, top_k = TOP_K):
    chunks = pipeline.chunks
    bm25 = pipeline.bm25
    client_q = pipeline.client_q
    # Bm25
    bm25_scores = bm25.get_scores(query.split())
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i:bm25_scores[i], reverse=True)[:top_k]
    # Dense
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype('float32')
    dense_results = client_q.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        limit=top_k,
        with_payload=True
    ).points

    dense_scores = {
        (r.payload["doc_id"], r.payload["chunk_id"]): r.score
        for r in dense_results
    }
    bm25_scores_dict = {
        (chunks[i].doc_id, chunks[i].chunk_id): bm25_scores[i]
        for i in bm25_top
    }
    all_keys = set(dense_scores) | set(bm25_scores_dict)
    fused = []
    for key in all_keys:
        score = ALPHA*dense_scores.get(key, 0) + (1-ALPHA)*bm25_scores_dict.get(key, 0)
        fused.append((score, key))
    fused.sort(reverse=True)
    top_keys = fused[:top_k]

    candidates = [
        next(c for c in chunks if (c.doc_id, c.chunk_id) == key)
        for _, key in top_keys
    ]

    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip(scores, candidates))
    scored.sort(reverse=True)

    return [c for _, c in scored[:top_k]]

def retrieve_with_context(pipeline: RAGPipeline, query: str, top_k=TOP_K):
    retrieved = retrieval(pipeline=pipeline, query=query, top_k=top_k)
    contexts = [c.text for c in retrieved]
    return retrieved, contexts

# ==========================
# Generate
# ==========================

def generate_answer(pipeline: RAGPipeline, query: str):

    logger.start_query(query=query)
    logger.start_timer("retrieval")

    retrieved = retrieval(pipeline=pipeline, query=query)
    logger.stop_timer("retrieval")
    dense_hits = [
        {"doc_id":c.doc_id, "chunk_id":c.chunk_id}
        for c in retrieved
    ]
    logger.log_retrieval(dense_hits=dense_hits)
    contexts = [
        f"[{c.source.name}#chunk_{c.chunk_id}]\n{c.text}"
        for c in retrieved
    ]
    context = "\n\n".join(contexts)
    
    logger.start_timer("llm")
    prompt = f"""
You are a helpful assistant.
Answer using ONLY the context.
If unsure, say: I don't know.

Question: {query}

Context:
{context}

Answer:
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role":"user", "content":prompt}],
        temperature=0.2
    )

    logger.stop_timer("llm")
    logger.log_llm(
        model="meta-llama/llama-3-8b-instruct",
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )

    logger.end_query()

    return response.choices[0].message.content

# ==========================
# CLI Entry
# ==========================

def main():
    pipeline = build_pipeline()
    if len(sys.argv) < 2:
        print("Please provide a query.")
        return
    query = sys.argv[1]
    answer = generate_answer(pipeline=pipeline, query=query)
    print("\nAnswer:\n")
    print(answer)

if __name__ == "__main__":
    main()


