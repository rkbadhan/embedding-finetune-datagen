"""
Synthetic Data Generation Pipeline for Embedding Model Training
================================================================

Replicates the two-stage persona-based query synthesis approach used by:
- Qwen3-Embedding (Zhang et al., 2025) 
- pplx-embed (Perplexity, 2025)

Pipeline Overview:
  1. Load documents from FineWeb-Edu (HuggingFace streaming)
  2. Load personas from PersonaHub (Ge et al., 2024)
  3. Embed all personas into a FAISS index (one-time cost)
  4. For each document, embed it and retrieve top-5 relevant personas via ANN
  5. [Future] Stage 1 LLM call: select persona + configure query type/difficulty
  6. [Future] Stage 2 LLM call: generate the actual query

This script implements steps 1-4. Steps 5-6 require an LLM endpoint.

References:
  - PersonaHub: https://huggingface.co/datasets/proj-persona/PersonaHub
  - FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  - Qwen3-Embedding paper: arXiv:2506.05176v3
  - pplx-embed paper: arXiv:2602.11151v2
"""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Central configuration for the synthesis pipeline."""

    # -- Document source --
    # FineWeb-Edu is the English source used by pplx-embed.
    # For Qwen3's multilingual variant you'd swap to their pretraining corpus.
    fineweb_dataset: str = "HuggingFaceFW/fineweb-edu"
    fineweb_split: str = "train"
    fineweb_sample_size: int = 1000       # number of docs to process (for learning/dev)
    fineweb_streaming: bool = True         # stream to avoid downloading 10TB+

    # -- Persona source --
    persona_dataset: str = "proj-persona/PersonaHub"
    persona_split: str = "train"
    persona_subset: str = "persona"        # PersonaHub has multiple configs
    persona_max_load: int = 50_000         # cap for dev; full set is ~200k

    # -- Embedding model for persona retrieval --
    # Using a small, fast model for the ANN index.
    # In production, Qwen3/pplx used their own retrieval models.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 256
    embedding_dim: int = 384               # matches all-MiniLM-L6-v2

    # -- Retrieval --
    top_k_personas: int = 5                # papers specify top-5
    faiss_nlist: int = 100                 # IVF clusters (tune for larger corpora)
    faiss_nprobe: int = 10                 # clusters to search at query time

    # -- Output --
    output_dir: str = "./synth_pipeline_output"
    save_every: int = 100                  # checkpoint every N documents


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("synth_pipeline")


# ---------------------------------------------------------------------------
# Step 1: Load Documents from FineWeb-Edu
# ---------------------------------------------------------------------------

def load_fineweb_documents(cfg: PipelineConfig) -> Iterator[dict]:
    """
    Stream documents from FineWeb-Edu.
    
    FineWeb-Edu is a 1.3T token, high-quality educational subset of FineWeb.
    pplx-embed used FineWeb-Edu for 50% of their English pretraining data
    (see Table 10 in arXiv:2602.11151v2).
    
    Each yielded dict has keys: 'text', 'url', 'id'
    
    Why streaming?
    - Full FineWeb-Edu is ~10TB. Streaming lets us pull documents on-demand
      without downloading the entire thing.
    - For a learning pipeline, we just need a manageable sample.
    """
    from datasets import load_dataset

    log.info(f"Loading FineWeb-Edu (streaming={cfg.fineweb_streaming})...")
    log.info(f"  Dataset: {cfg.fineweb_dataset}")
    log.info(f"  Sample size: {cfg.fineweb_sample_size}")

    ds = load_dataset(
        cfg.fineweb_dataset,
        split=cfg.fineweb_split,
        streaming=cfg.fineweb_streaming,
    )

    count = 0
    for example in ds:
        if count >= cfg.fineweb_sample_size:
            break

        text = example.get("text", "")

        # Basic quality filter: skip very short documents.
        # In production you'd apply more sophisticated filtering
        # (language detection, dedup, content quality scoring).
        if len(text.split()) < 50:
            continue

        yield {
            "id": example.get("id", f"doc_{count}"),
            "url": example.get("url", ""),
            "text": text,
        }
        count += 1

    log.info(f"Loaded {count} documents from FineWeb-Edu.")


# ---------------------------------------------------------------------------
# Step 2: Load Personas from PersonaHub
# ---------------------------------------------------------------------------

def load_persona_hub(cfg: PipelineConfig) -> list[dict]:
    """
    Load persona descriptions from PersonaHub (Ge et al., 2024).
    
    PersonaHub contains ~200k diverse persona descriptions, each representing
    a synthetic "user" with specific expertise, interests, and background.
    
    Both Qwen3 and pplx-embed retrieve the top-5 most relevant personas for
    each document, then use them to diversify the synthetic queries. The idea
    is that a "marine biologist" would ask very different questions about an
    ocean document than a "high school student" would.
    
    Dataset structure:
      - 'input persona': the persona description string
      - 'synthesized text': example text the persona might produce
      - 'description': additional context
    
    We use the 'input persona' field as the persona text to embed.
    """
    from datasets import load_dataset

    log.info(f"Loading PersonaHub personas...")
    log.info(f"  Dataset: {cfg.persona_dataset} (config={cfg.persona_subset})")
    log.info(f"  Max personas: {cfg.persona_max_load}")

    ds = load_dataset(
        cfg.persona_dataset,
        cfg.persona_subset,
        split=cfg.persona_split,
    )

    personas = []
    for i, example in enumerate(ds):
        if i >= cfg.persona_max_load:
            break

        persona_text = example.get("input persona", "")
        if not persona_text or len(persona_text.strip()) < 10:
            continue

        personas.append({
            "id": f"persona_{i}",
            "text": persona_text.strip(),
            # Keep the synthesized text for potential downstream use
            "synthesized_text": example.get("synthesized text", ""),
            "description": example.get("description", ""),
        })

    log.info(f"Loaded {len(personas)} personas from PersonaHub.")
    return personas


# ---------------------------------------------------------------------------
# Step 3: Build Embedding Index for Personas
# ---------------------------------------------------------------------------

class PersonaRetriever:
    """
    Embeds all personas and builds a FAISS index for fast ANN retrieval.
    
    Architecture decisions:
    
    1. We embed personas ONCE and store them in a FAISS IVF index.
       This is the same pattern used in production: the persona "library"
       is static, so we pre-compute all embeddings.
    
    2. For each incoming document, we embed it with the same model and
       query the index for top-5 nearest personas.
    
    3. The embedding model here (all-MiniLM-L6-v2) is a lightweight choice
       for learning purposes. In production, Qwen3 used their own retrieval
       model, and pplx-embed likely used a similar internal model.
    
    Why FAISS IVF instead of brute-force?
    - With 200k personas, brute-force is fast enough (<10ms per query).
    - But the IVF pattern is what you'd use at scale (millions of personas
      or if you're doing this for every document in a 150M-doc corpus).
    - Good to learn the pattern now.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model = None
        self.index = None
        self.personas: list[dict] = []
        self._persona_embeddings: np.ndarray | None = None

    def load_model(self):
        """Load the embedding model."""
        from sentence_transformers import SentenceTransformer

        log.info(f"Loading embedding model: {self.cfg.embedding_model}")
        self.model = SentenceTransformer(self.cfg.embedding_model)
        log.info(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def build_index(self, personas: list[dict]):
        """
        Embed all personas and build FAISS index.
        
        Steps:
          1. Extract persona text strings
          2. Batch-encode with sentence-transformers
          3. Normalize embeddings (for cosine similarity via inner product)
          4. Build FAISS IVF index
          5. Train and add vectors
        """
        import faiss

        self.personas = personas
        persona_texts = [p["text"] for p in personas]

        log.info(f"Encoding {len(persona_texts)} personas...")
        t0 = time.time()

        # Batch encode all persona descriptions
        embeddings = self.model.encode(
            persona_texts,
            batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )
        self._persona_embeddings = np.array(embeddings, dtype=np.float32)

        elapsed = time.time() - t0
        log.info(f"  Encoded in {elapsed:.1f}s "
                 f"({len(persona_texts)/elapsed:.0f} personas/sec)")

        # Build FAISS index
        dim = self._persona_embeddings.shape[1]
        n_personas = len(persona_texts)

        if n_personas < 1000:
            # For small datasets, just use flat (brute-force) index
            log.info("  Using Flat index (small dataset)")
            self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine on normalized vectors
        else:
            # IVF index for larger persona sets
            nlist = min(self.cfg.faiss_nlist, n_personas // 10)
            log.info(f"  Building IVF index: {nlist} clusters, dim={dim}")

            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = self.cfg.faiss_nprobe

            # IVF requires training on the data distribution
            log.info("  Training FAISS index...")
            self.index.train(self._persona_embeddings)

        self.index.add(self._persona_embeddings)
        log.info(f"  FAISS index ready: {self.index.ntotal} vectors")

    def retrieve(self, document_text: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve top-k most relevant personas for a given document.
        
        This is the core operation that both Qwen3 and pplx-embed perform:
        "The candidate characters are retrieved from Persona Hub, selecting
         the top five most relevant to the given document."
        
        Args:
            document_text: The document to find personas for.
            top_k: Number of personas to retrieve (default: config value).
        
        Returns:
            List of dicts with persona info + similarity score.
        """
        if top_k is None:
            top_k = self.cfg.top_k_personas

        # Encode the document
        doc_embedding = self.model.encode(
            [document_text],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search the FAISS index
        scores, indices = self.index.search(doc_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for unfilled slots
                continue
            persona = self.personas[idx].copy()
            persona["similarity_score"] = float(score)
            results.append(persona)

        return results

    def retrieve_batch(self, documents: list[str], top_k: int | None = None) -> list[list[dict]]:
        """
        Batch retrieval for multiple documents at once.
        
        Much faster than calling retrieve() in a loop because:
        1. Embedding model processes all docs in one batch
        2. FAISS searches all queries in parallel
        """
        if top_k is None:
            top_k = self.cfg.top_k_personas

        # Batch encode all documents
        doc_embeddings = self.model.encode(
            documents,
            batch_size=self.cfg.embedding_batch_size,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Batch search
        scores, indices = self.index.search(doc_embeddings, top_k)

        all_results = []
        for doc_scores, doc_indices in zip(scores, indices):
            results = []
            for score, idx in zip(doc_scores, doc_indices):
                if idx == -1:
                    continue
                persona = self.personas[idx].copy()
                persona["similarity_score"] = float(score)
                results.append(persona)
            all_results.append(results)

        return all_results


# ---------------------------------------------------------------------------
# Step 4: Putting It All Together
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig | None = None):
    """
    Run the full document -> persona retrieval pipeline.
    
    This produces output like:
    
    {
        "document_id": "doc_42",
        "document_text": "The ocean's twilight zone...",
        "document_url": "https://...",
        "top_5_personas": [
            {
                "id": "persona_1234",
                "text": "A marine biologist specializing in deep-sea ecosystems...",
                "similarity_score": 0.82
            },
            ...
        ]
    }
    
    This output is the input to Stage 1 (LLM configuration) and Stage 2
    (LLM query generation) of the synthesis pipeline.
    """
    if cfg is None:
        cfg = PipelineConfig()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load personas and build retrieval index ---
    personas = load_persona_hub(cfg)
    
    retriever = PersonaRetriever(cfg)
    retriever.load_model()
    retriever.build_index(personas)

    # --- Process documents ---
    results = []
    doc_batch = []
    doc_meta_batch = []
    batch_size = 32  # process docs in mini-batches for efficiency

    log.info("=" * 60)
    log.info("Starting document processing...")
    log.info("=" * 60)

    t0 = time.time()

    for doc in load_fineweb_documents(cfg):
        doc_batch.append(doc["text"])
        doc_meta_batch.append(doc)

        if len(doc_batch) >= batch_size:
            # Batch retrieve personas
            batch_personas = retriever.retrieve_batch(doc_batch)

            for meta, personas_for_doc in zip(doc_meta_batch, batch_personas):
                result = {
                    "document_id": meta["id"],
                    "document_url": meta["url"],
                    "document_text_preview": meta["text"][:500],  # preview only for output
                    "document_word_count": len(meta["text"].split()),
                    "top_5_personas": [
                        {
                            "persona_id": p["id"],
                            "persona_text": p["text"],
                            "similarity_score": round(p["similarity_score"], 4),
                        }
                        for p in personas_for_doc
                    ],
                }
                results.append(result)

            doc_batch = []
            doc_meta_batch = []

            # Progress logging
            if len(results) % cfg.save_every == 0:
                elapsed = time.time() - t0
                rate = len(results) / elapsed
                log.info(f"  Processed {len(results)} docs ({rate:.1f} docs/sec)")

    # Process remaining docs in the last partial batch
    if doc_batch:
        batch_personas = retriever.retrieve_batch(doc_batch)
        for meta, personas_for_doc in zip(doc_meta_batch, batch_personas):
            result = {
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
            }
            results.append(result)

    total_time = time.time() - t0
    log.info("=" * 60)
    log.info(f"Pipeline complete: {len(results)} documents processed in {total_time:.1f}s")
    log.info(f"  Average: {len(results)/total_time:.1f} docs/sec")
    log.info("=" * 60)

    # --- Save results ---
    output_path = output_dir / "doc_persona_pairs.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Saved results to {output_path}")

    # --- Print a few examples ---
    log.info("\n" + "=" * 60)
    log.info("EXAMPLE OUTPUTS")
    log.info("=" * 60)

    for i, r in enumerate(results[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"ID:    {r['document_id']}")
        print(f"Words: {r['document_word_count']}")
        print(f"Text:  {r['document_text_preview'][:200]}...")
        print(f"\nTop-5 Personas:")
        for j, p in enumerate(r["top_5_personas"]):
            print(f"  {j+1}. [sim={p['similarity_score']:.4f}] {p['persona_text'][:100]}...")

    return results


# ---------------------------------------------------------------------------
# Stage 1 & 2 Templates (for reference / future implementation)
# ---------------------------------------------------------------------------

# These are the exact templates from the Qwen3-Embedding paper (Appendix A.1).
# You would call these with an LLM API (e.g., Qwen3-32B, GPT-4, Claude).

STAGE_1_CONFIG_TEMPLATE = """Given a **Passage** and **Character**, select the appropriate option from \
three fields: Character, Question_Type, Difficulty, and return the output in JSON format.

First, select the Character who are likely to be interested in the Passage from the candidates. \
Then select the Question_Type that the Character might ask about the Passage; \
Finally, choose the Difficulty of the possible question based on the Passage, the Character, \
and the Question_Type.

Character: Given by input **Character**
Question_Type:
- keywords: The query focuses on specific keywords or key phrases from the passage.
- acquire_knowledge: The query seeks to learn or understand information from the passage.
- summary: The query asks for a summary or overview of the passage content.
- yes_or_no: The query can be answered with a yes or no based on the passage.
- background: The query asks about background context related to the passage.
Difficulty:
- high_school: Suitable for a high school student level.
- university: Suitable for a university student level.
- phd: Suitable for a PhD researcher level.

Now, generate the **output** based on the **Passage** and **Character** from user, \
the **Passage** will be in {language} language and the **Character** will be in English.
Ensure to generate only the JSON output with content in English.

**Passage**:
{passage}

**Character**:
{character_candidates}
"""

STAGE_2_QUERY_TEMPLATE = """Given a **Character**, **Passage**, and **Requirement**, generate a query from \
the **Character**'s perspective that satisfies the **Requirement** and can be used to retrieve the **Passage**. \
Please return the result in JSON format.

Here is an example:
<example>

Now, generate the **output** based on the **Character**, **Passage** and **Requirement** from user, \
the **Passage** will be in {corpus_language} language, the **Character** and **Requirement** will be in English.
Ensure to generate only the JSON output, with the key in English and the value in {queries_language} language.

**Character**:
{character}

**Passage**:
{passage}

**Requirement**:
- Type: {type};
- Difficulty: {difficulty};
- Length: the length of the generated sentences should be {length} words;
- Language: the language in which the results are generated should be {language} language;
"""


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For learning: small sample, fast iteration
    config = PipelineConfig(
        fineweb_sample_size=100,       # start small
        persona_max_load=10_000,       # subset of PersonaHub
        output_dir="./synth_pipeline_output",
    )

    results = run_pipeline(config)
