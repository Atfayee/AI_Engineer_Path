from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import sys
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
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
        if p.suffix.lower() == '.md':
            docs.append({
                "doc_id": p.name,
                "text": p.read_text(encoding="utf-8", errors="ignore"),
                "source": p,
                "last_edited": p.stat().st_mtime
            })
    return docs

def clean_text(text: str) -> str:
    t = text.replace("\r\n", "\n")
    t = "\n".join([l for l in t.splitlines()])
    while "\n\n\n" in t:
        t = t.replace("\n\n\n", "\n\n")
    return t


def chunk_text(d: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Chunk]:
    chunks = []
    i = 0
    c_id = 0
    text = d["text"]
    while i < len(text):
        j = min(len(text), i + chunk_size)
        text_chunk = text[i:j]
        chunks.append(Chunk(
            chunk_id=c_id,
            doc_id=d["doc_id"],
            text=text_chunk,
            source=d["source"],
            last_edited=d["last_edited"]
        ))
        c_id += 1
        i = max(i+1, j-chunk_overlap)
    return chunks
def embed_texts(texts: list[str], embed_model: SentenceTransformer):
    return embed_model.encode(texts, normalize_embeddings=True).astype("float32")

def embed_query(text: str, embed_model: SentenceTransformer):
    return embed_texts(texts=[text], embed_model=embed_model)

def call_llm(prompt: str):
    response = client.chat.completions.create(
        model="meta-llama/llama-3-8b-instruct",
        messages=[
            {"role":"user", "content":prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content
 
def main():
    base_path = Path(__file__).parent
    data_dir = base_path / "data"
    docs = load_docs(data_dir=data_dir)

    chunks: list[Chunk] = []
    for d in docs:
        d["text"] = clean_text(d["text"])
        chunk = chunk_text(d)
        chunks.extend(chunk)
    texts = [c.text for c in chunks]
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_texts(texts=texts, embed_model=embed_model)
    dim = embeddings.shape[1]
    # print(f"Vector Dim: {dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    # print(f"Index size: {index.ntotal}")
    if len(sys.argv) < 2:
        print("Please add a query")
        return
    query = sys.argv[1]
    # print(f"query: {query}")
    query_vec = embed_query(query, embed_model=embed_model)
    scores, indices = index.search(query_vec, k=5)
    retrieved = []
    for rank, idx in enumerate(indices[0]):
        retrieved.append({
            "score": float(scores[0][rank]),
            "chunk": chunks[idx]
        })
    if retrieved[0]["score"] < 0.4:
        print("Low confidence retrival. Refusing.")
        return
    # print(retrieved)
    contexts = []
    for item in retrieved:
        chunk = item["chunk"]
        block = f"[{chunk.source}#chunk_{chunk.chunk_id}\n{chunk.text}"
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