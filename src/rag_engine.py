"""
Horae - RAG Evidence Retrieval Engine

Phase 1:
    Merchant policy documents
        ↓
    Chunking
        ↓
    Sentence Transformer embeddings
        ↓
    FAISS vector index
        ↓
    Relevant evidence retrieval

LLM generation will be added in Phase 2.
"""

from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
RAG_STORAGE_DIR = PROJECT_ROOT / "models" / "rag"

INDEX_PATH = RAG_STORAGE_DIR / "policy.index"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_policy_documents() -> List[Dict]:
    """
    Load all .txt policy documents from the knowledge base.
    """

    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge base not found:\n{KNOWLEDGE_BASE_DIR}"
        )

    documents = []

    for file_path in sorted(
        KNOWLEDGE_BASE_DIR.glob("*.txt")
    ):
        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    if not documents:
        raise ValueError(
            "No policy documents found in knowledge_base/"
        )

    print(
        f"📚 Loaded {len(documents)} policy documents."
    )

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Section-aware policy chunking.

    Keeps policy sections together so retrieved evidence
    remains readable and traceable to the original policy.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks = []
    current_section = []
    current_length = 0

    for line in lines:

        # Detect numbered policy sections such as:
        # "1. GENERAL REFUND ELIGIBILITY"
        # "2. ITEM NOT RECEIVED"
        is_section_heading = (
            len(line) > 2
            and line[0].isdigit()
            and "." in line[:3]
        )

        # If we encounter a new section,
        # finalize the previous section.
        if is_section_heading and current_section:

            section_text = "\n".join(current_section)

            chunks.append(section_text)

            current_section = []
            current_length = 0

        current_section.append(line)
        current_length += len(line)

        # Safety fallback for unusually large sections.
        if current_length >= chunk_size:

            section_text = "\n".join(current_section)

            chunks.append(section_text)

            # Preserve a small textual overlap.
            overlap_text = section_text[-overlap:]

            current_section = [overlap_text]
            current_length = len(overlap_text)

    # Add final section.
    if current_section:
        chunks.append(
            "\n".join(current_section)
        )

    return chunks


def build_chunks(
    documents: List[Dict],
) -> List[Dict]:
    """
    Convert policy documents into searchable chunks.
    """

    chunks = []

    for document in documents:

        text_chunks = chunk_text(
            document["text"]
        )

        for index, chunk in enumerate(
            text_chunks
        ):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "source": document["source"],
                    "section_index": index,
                    "text": chunk,
                }
            )

    print(
        f"🧩 Created {len(chunks)} searchable policy chunks."
    )

    return chunks


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Load the sentence-transformer embedding model.
    """

    print(
        f"🧠 Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("✅ Embedding model loaded.")

    return model


# ============================================================
# VECTOR INDEX
# ============================================================

def build_faiss_index(
    chunks: List[Dict],
    embedding_model,
):
    """
    Generate embeddings and create a FAISS cosine-similarity
    index using normalized vectors.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"🔢 Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    # Normalize vectors so inner product ≈ cosine similarity
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    print(
        f"✅ FAISS index created "
        f"({index.ntotal} vectors)."
    )

    return index


# ============================================================
# SAVE / LOAD INDEX
# ============================================================

def save_index(
    index,
    chunks: List[Dict],
):
    """
    Save the FAISS index and its metadata.
    """

    RAG_STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    metadata_path = (
        RAG_STORAGE_DIR
        / "chunks.npy"
    )

    np.save(
        metadata_path,
        np.array(
            chunks,
            dtype=object,
        ),
        allow_pickle=True,
    )

    print(
        f"💾 FAISS index saved to:\n"
        f"   {INDEX_PATH}"
    )

    print(
        f"💾 Chunk metadata saved to:\n"
        f"   {metadata_path}"
    )


def load_index():
    """
    Load previously built FAISS index and chunk metadata.
    """

    metadata_path = (
        RAG_STORAGE_DIR
        / "chunks.npy"
    )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index does not exist. "
            "Run build_rag_index() first."
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    chunks = np.load(
        metadata_path,
        allow_pickle=True,
    ).tolist()

    return index, chunks


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(
    query: str,
    embedding_model,
    index,
    chunks: List[Dict],
    top_k: int = 3,
) -> List[Dict]:
    """
    Retrieve the most relevant policy clauses for a query.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype(np.float32)

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "section_index": chunk[
                    "section_index"
                ],
                "text": chunk["text"],
            }
        )

    return results


# ============================================================
# BUILD COMPLETE RAG INDEX
# ============================================================

def build_rag_index():
    """
    Complete indexing pipeline.
    """

    print("\n" + "=" * 70)
    print("🔎 HORAЕ RAG INDEX BUILD")
    print("=" * 70)

    documents = load_policy_documents()

    chunks = build_chunks(
        documents
    )

    embedding_model = load_embedding_model()

    index = build_faiss_index(
        chunks,
        embedding_model,
    )

    save_index(
        index,
        chunks,
    )

    print("\n" + "=" * 70)
    print("🎉 RAG INDEX READY")
    print("=" * 70)

    return (
        embedding_model,
        index,
        chunks,
    )


# ============================================================
# TEST RETRIEVAL
# ============================================================

def test_retrieval():
    """
    Run a simple retrieval test.
    """

    print("\n" + "=" * 70)
    print("🧪 RAG RETRIEVAL TEST")
    print("=" * 70)

    embedding_model, index, chunks = (
        build_rag_index()
    )

    test_queries = [
        "Customer claims that their package was not received.",
        "Customer says the product was defective.",
        "Customer claims the transaction was unauthorized.",
    ]

    for query in test_queries:

        print("\n" + "-" * 70)
        print(f"QUERY: {query}")
        print("-" * 70)

        results = retrieve_evidence(
            query=query,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
            top_k=3,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"{result['source']} "
                f"(similarity: "
                f"{result['score']:.3f})"
            )

            print(
                result["text"][:500]
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_retrieval()