import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

# Fix: Set multi-threading environment limits *before* any sub-process or threading starts 
# to protect against macOS Accelerated frameworks interfering with multiprocessing. 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Classes & Definitions ---

@dataclasses.dataclass
class PipelineConfig:
    fineweb_dataset: str = "HuggingFaceFW/fineweb-edu"
    fineweb_split: str = "train"
    fineweb_sample_size: int = 1000 
    fineweb_streaming: bool = True   
    persona_dataset: str = "proj-persona/PersonaHub"
    persona_split: str = "train"
    persona_subset: str = "persona"
    persona_max_load: int = 50_000
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_batch_size: int = 64
    embedding_dim: int = 768               
    embedding_prefix_document: str = "search_document: " 
    embedding_prefix_query: str = "search_query: "        
    top_k_personas: int = 5                
    faiss_nlist: int = 100                 
    faiss_nprobe: int = 10                 
    output_dir: str = "./synth_pipeline_output"
    save_every: int = 100

def load_fineweb_documents(cfg: PipelineConfig):
    from datasets import load_dataset
    logger.info(f"Loading FineWeb-Edu (streaming={cfg.fineweb_streaming})...")
    
    ds = load_dataset(cfg.fineweb_dataset, split=cfg.fineweb_split, streaming=cfg.fineweb_streaming)

    count = 0
    for example in ds:
        if count >= cfg.fineweb_sample_size:
            break
        text = example.get("text", "")
        if len(text.split()) < 50:
            continue

        yield {
            "id": example.get("id", f"doc_{count}"),
            "url": example.get("url", ""),
            "text": text,
        }
        count += 1

def load_persona_hub(cfg: PipelineConfig) -> list[dict]:
    from datasets import load_dataset
    logger.info(f"Loading PersonaHub personas...")
    ds = load_dataset(cfg.persona_dataset, cfg.persona_subset, split=cfg.persona_split, streaming=True)

    persona_field = "persona" if cfg.persona_subset == "persona" else "input_persona"
    personas = []
    for i, example in enumerate(ds):
        if len(personas) >= cfg.persona_max_load:
            break
        persona_text = example.get(persona_field, "")
        if not persona_text or len(persona_text.strip()) < 10:
            continue
        personas.append({
            "id": f"persona_{i}",
            "text": persona_text.strip(),
            "synthesized_text": example.get("synthesized_text", ""),
            "description": example.get("description", ""),
        })

    logger.info(f"Loaded {len(personas)} personas.")
    return personas

class PersonaRetriever:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model = None
        self.index = None
        self.personas: list[dict] = []
        self._persona_embeddings: np.ndarray | None = None

    def load_model(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {self.cfg.embedding_model}")
        self.model = SentenceTransformer(self.cfg.embedding_model, trust_remote_code=True)
        # --- FIX #5: Avoid OOM by restricting processing length per query ---
        self.model.max_seq_length = 1024
        
    def build_index(self, personas: list[dict]):
        import faiss
        
        # Enforce 1 thread for FAISS to stop segfaults on Mac!
        faiss.omp_set_num_threads(1)

        self.personas = personas
        prefix = self.cfg.embedding_prefix_document
        persona_texts = [prefix + p["text"] for p in personas]

        logger.info(f"Encoding {len(persona_texts)} personas...")
        embeddings = self.model.encode(
            persona_texts,
            batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        # --- FIX #3: FAISS required C-contiguous array ---
        self._persona_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # --- FIX #4: Replace NaNs from MPS backend failures before FAISS errors out ---
        if not np.all(np.isfinite(self._persona_embeddings)):
            n_bad = int(np.sum(~np.isfinite(self._persona_embeddings)))
            logger.warning(f"Found {n_bad} non-finite values in embeddings, replacing with zeros")
            self._persona_embeddings = np.nan_to_num(self._persona_embeddings)
            norms = np.linalg.norm(self._persona_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._persona_embeddings = np.ascontiguousarray(
                self._persona_embeddings / norms, dtype=np.float32
            )

        dim = self._persona_embeddings.shape[1]
        n_personas = len(persona_texts)

        if n_personas < 1000:
            self.index = faiss.IndexFlatIP(dim)
        else:
            nlist = min(self.cfg.faiss_nlist, n_personas // 10)
            logger.info(f"  Building IVF index: {nlist} clusters, dim={dim}")
            
            # --- FIX #1: Guard against Python GC destroying the quantizer ---
            self._quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(self._quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = self.cfg.faiss_nprobe

            logger.info("  Training FAISS index...")
            self.index.train(self._persona_embeddings)

        self.index.add(self._persona_embeddings)
        logger.info(f"  FAISS index ready: {self.index.ntotal} vectors")

    def retrieve_batch(self, documents: list[str], top_k: int | None = None) -> list[list[dict]]:
        if top_k is None: top_k = self.cfg.top_k_personas
        prefix = self.cfg.embedding_prefix_query
        prefixed_docs = [prefix + d for d in documents]
        
        doc_embeddings = self.model.encode(
            prefixed_docs,
            batch_size=self.cfg.embedding_batch_size,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(doc_embeddings, top_k)
        
        all_results = []
        for doc_scores, doc_indices in zip(scores, indices):
            results = []
            for score, idx in zip(doc_scores, doc_indices):
                if idx == -1: continue
                persona = self.personas[idx].copy()
                persona["similarity_score"] = float(score)
                results.append(persona)
            all_results.append(results)

        return all_results

def run_pipeline(cfg: PipelineConfig | None = None):
    if cfg is None:
        cfg = PipelineConfig()

    import time
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    personas = load_persona_hub(cfg)
    
    retriever = PersonaRetriever(cfg)
    retriever.load_model()
    retriever.build_index(personas)

    # --- FIX #2: Convert generator to list BEFORE calling model.encode to prevent deadlocks ---
    documents = list(load_fineweb_documents(cfg))

    output_path = output_dir / "doc_persona_pairs.jsonl"
    doc_batch = []
    doc_meta_batch = []
    batch_size = 16 
    next_log_at = cfg.save_every
    total_processed = 0

    def _process_batch(doc_texts, doc_metas):
        batch_personas = retriever.retrieve_batch(doc_texts)
        batch_results = []
        for meta, personas_for_doc in zip(doc_metas, batch_personas):
            batch_results.append({
                "document_id": meta["id"],
                "document_url": meta["url"],
                "document_text_preview": meta["text"][:500],
                "document_word_count": len(meta["text"].split()),
                "top_5_personas": [
                    {
                        "persona_id": p["id"],
                        "persona_text": p["text"],
                        "similarity_score": round(p["similarity_score"], 4),
                    }
                    for p in personas_for_doc
                ],
            })
        return batch_results

    logger.info("=" * 60)
    logger.info("Starting document processing...")
    logger.info("=" * 60)
    t0 = time.time()

    # --- FIX #6: Write chunked data out to prevent Out Of Memory crashes on larger scale datasets ---
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            doc_batch.append(doc["text"])
            doc_meta_batch.append(doc)

            if len(doc_batch) >= batch_size:
                batch_results = _process_batch(doc_batch, doc_meta_batch)
                
                for r in batch_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
                total_processed += len(batch_results)
                doc_batch = []
                doc_meta_batch = []

                if total_processed >= next_log_at:
                    elapsed = time.time() - t0
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    logger.info(f"  Processed {total_processed} docs ({rate:.1f} docs/sec)")
                    next_log_at += cfg.save_every

        if doc_batch:
            batch_results = _process_batch(doc_batch, doc_meta_batch)
            for r in batch_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_processed += len(batch_results)

    total_time = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Pipeline complete: {total_processed} documents processed in {total_time:.1f}s")
    logger.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    config = PipelineConfig(
        fineweb_sample_size=100,      
        persona_max_load=10_000,       
        output_dir="./synth_pipeline_output",
    )
    run_pipeline(config)
