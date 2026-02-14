from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import sys, os, uuid
from openai import OpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


@dataclass
class Chunk:
    chunk_id: int
    doc_id: int
    text: str
    source: str
    last_edited: str


def load_docs(data_dir: Path):
    docs = []
    for p in Path(data_dir).glob("**/*"):
        if p.suffix.lower() == ".md":
            docs.append(
                {
                    "doc_id": p.name,
                    "text": p.read_text(encoding="utf-8", errors="ignore"),
                    "source": p,
                    "last_edited": p.stat().st_mtime,
                }
            )
    return docs


def clean_text(text: str) -> str:
    t = text.replace("\r\n", "\n")
    t = "\n".join([l for l in t.splitlines()])
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t


def embed_texts(texts: list[str], embed_model: SentenceTransformer):
    return embed_model.encode(texts, normalize_embeddings=True).astype("float32")


def embed_query(text: str, embed_model: SentenceTransformer):
    return embed_texts(texts=[text], embed_model=embed_model)


def call_llm(prompt: str):
    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# Split into sections
# merge paragraphs in each section by maxsize
# if paragraph is too long, split it by maxsize characters
# Section-aware + paragraph recursive + char fallback


def split_by_headers(text: str) -> list[str]:
    sections = re.split(r"?=^#{1,6} ", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def recursive_chunk(section: str, max_size=1000) -> list[str]:
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
                    chunks.append(p[i : i + max_size])
                current = ""
            else:
                current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


def chunk_text(d: dict, max_size=1000) -> list[Chunk]:
    sections = split_by_headers(d["text"])
    chunks = []
    c_id = 0

    for section in sections:
        sub_chunks = recursive_chunk(section=section, max_size=max_size)
        for sc in sub_chunks:
            chunks.append(
                Chunk(
                    chunk_id=c_id,
                    doc_id=d["doc_id"],
                    text=sc,
                    source=d["source"],
                    last_edited=d["last_edited"],
                )
            )
            c_id += 1
    return chunks


def expand_neighbors(result_points: list[models.PointStruct], doc_chunk_map, window=1):
    expanded = []
    seen = set()

    for r in result_points:
        payload = r.payload
        doc_id = payload["doc_id"]
        chunk_id = payload["chunk_id"]
        doc_chunks = doc_chunk_map[doc_id]
        for idx, c in enumerate(doc_chunks):
            if c.chunk_id == chunk_id:
                break
        start = max(0, idx - window)
        end = min(len(doc_chunks), idx + window)
        for i in range(start, end):
            c = doc_chunks[i]
            key = (c.doc_id, c.chunk_id)
            if key not in seen:
                expanded.append(c)
                seen.add(key)
        return expanded


# metadata filter
SOURCE = "Llama.md"
TOP_K = 5
COLLECTION = "rag_v1"


def main():
    base_path = Path(__file__).parent
    data_dir = base_path / "data"
    docs = load_docs(data_dir=data_dir)

    chunks: list[Chunk] = []
    for d in docs:
        d["text"] = clean_text(d["text"])
        chunk = chunk_text(d)
        chunks.extend(chunk)
    # chunk neighbour expansion
    doc_chunk_map = defaultdict(list[Chunk])
    for c in chunks:
        doc_chunk_map[c.doc_id].append(c)
    for doc_id in doc_chunk_map:
        doc_chunk_map[doc_id].sort(key=lambda x: x.chunk_id)

    texts = [c.text for c in chunks]
    # bm25 index
    tokenized_corpuse = [c.text.split() for c in texts]
    bm25 = BM25Okapi(tokenized_corpuse)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_texts(texts=texts, embed_model=embed_model)
    dim = embeddings.shape[1]

    client_q = QdrantClient(url="http://localhost:6333")
    client_q.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    points = []
    for i, (c, v) in enumerate(zip(chunks, embeddings)):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=v.tolist(),
                payload={
                    "doc_id": str(c.doc_id),
                    "chunk_id": int(c.chunk_id),
                    "source": c.source.name,
                    "last_edited": float(c.last_edited),
                    "text": c.text,
                },
            )
        )
    client_q.upsert(collection_name=COLLECTION, points=points)

    if len(sys.argv) < 2:
        print("Please add a query")
        return
    query = sys.argv[1]
    # print(f"query: {query}")

    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:TOP_K]

    query_vec = embed_query([query], embed_model=embed_model)

    qfilter = None
    if SOURCE:
        qfilter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source", match=models.MatchValue(value=SOURCE)
                )
            ]
        )
    dense_results = client_q.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        query_filter=qfilter,
        limit=TOP_K,
        with_payload=True,
    ).points
    if not dense_results:
        print("No matching documents found with given medata filter")
        return

    # Linear Fusion(Hybrid Search: Semantic + BM25)
    alpha = 0.6
    dense_scores = {
        (r.palyload["doc_id"], r.payload["chunk_id"]):r.score
        for r in dense_results
    }
    bm25_scores_dict = {
        (chunks[i].doc_id, chunks[i].chunk_id):bm25_scores[i]
        for i in bm25_top_indices
    }
    all_keys = set(dense_scores) | set(bm25_scores_dict)
    fused = []
    for key in all_keys:
        d_score = dense_scores.get(key, 0)
        b_score = bm25_scores_dict.get(key, 0)
        score = alpha * d_score + (1-alpha)*b_score
        fused.append((score, key))
    fused.sort(reverse=True)
    top_keys = fused[:TOP_K]

    hybrid_candidates = [
        next(c for c in chunks if (c.doc_id, c.chunk_id) == key)
        for _, key in top_keys
    ]

    expanded_chunks = expand_neighbors(
        result_points=[
            models.PointStruct(
                id=None,
                vector=None,
                payload={"doc_id": c.doc_id, "chunk_id": c.chunk_id},
            )
            for c in hybrid_candidates
        ],
        doc_chunk_map=doc_chunk_map,
        window=1,
    )
    contexts = []
    for c in expanded_chunks:
        block = f"[{c.source.name}#chunk_{c.chunk_id}]\n{c.text}"
        contexts.append(block)
    context = "\n\n".join(contexts)
    prompt = f"""
        You are a helpful QA assistant.
        Use ONLY the provieded context to answer the question.
        If the answer if not explicity contained in the context,
        response exactly with:

        I don't know.

        Rules:
        - Do not use outside knowledge.
        - Do not guess.
        - Cite sources using this format: [source#chunk_id]
        - Every factual claim must have a citation.

        Question: {query}
        Context: {context}
        Answer: 
    """
    answer = call_llm(prompt)
    print(answer)


if __name__ == "__main__":
    main()
