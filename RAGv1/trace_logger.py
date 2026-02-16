import json
import time
import uuid
from pathlib import Path

class TraceLogger:
    def __init__(self, log_path="logs/rag_trace.jsonl"):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.reset()
    
    def reset(self):
        self.trace = {
            "query_id": str(uuid.uuid4()),
            "query": None,
            "metadata_filter": None,
            "retrieval": {},
            "rerank": {},
            "llm": {},
            "timing": {},
            "metrics": {},
        }
        self._timers = {}
    
    def start_timer(self, name):
        self._timers[name] = time.time()

    def stop_timer(self, name):
        if name in self._timers:
            elapsed = (time.time() - self._timers[time])*1000
            self.trace["timing"][f"{name}_ms"] = round(elapsed, 2)

    def start_query(self, query, metadata_filter=None):
        self.reset()
        self.trace["query"] = query
        self.trace["metadata_filter"] = metadata_filter
        self.start_timer("total")

    def log_retrieval(self, dense_hits=None, bm25_hits=None, fused=None):
        self.trace["retrieval"] = {
            "dense_hits": dense_hits,
            "bm25_hits": bm25_hits,
            "fusion_scores": fused
        }

    def log_rerank(self, rerank_scores):
        self.trace["rerank"] = {
            "rerank_scores": rerank_scores
        }

    def log_llm(self, model, prompt_tokens=None, completion_tokens=None):
        self.trace["llm"] = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    def log_metrics(self, metrics_dict):
        self.trace["metrics"] = metrics_dict
    

    def end_query(self):
        self.stop_timer("total")
        self.save()

    def save(self):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.trace, ensure_ascii=False) + "\n")