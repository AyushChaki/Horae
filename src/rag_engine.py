"""
Horae — RAG Evidence Retrieval Engine
=====================================

RAG v2.1

Pipeline:
    Policy Documents
        ↓
    Section-aware Chunking
        ↓
    Sentence Transformer Embeddings
        ↓
    FAISS Semantic Retrieval
        ↓
    Dispute-aware Reranking
        ↓
    Relevance Filtering
        ↓
    Deduplication
        ↓
    Verified Policy Evidence

Design principles:
    - Semantic retrieval finds candidate evidence.
    - Dispute-aware reranking prioritizes the correct policy.
    - Relevance filtering removes unrelated evidence.
    - RAG never decides the final dispute outcome.
    - Evidence Engine remains responsible for risk/defense scoring.
    - Every result remains traceable to source + section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
METADATA_PATH = RAG_STORAGE_DIR / "chunks.npy"


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

DEFAULT_TOP_K = 3

# Retrieve more candidates before reranking/filtering.
RETRIEVAL_CANDIDATES = 8

# Final evidence threshold.
MIN_RERANK_SCORE = 0.40

# Threshold used for relevance labels.
HIGH_RELEVANCE_THRESHOLD = 0.70
MEDIUM_RELEVANCE_THRESHOLD = 0.50

# Maximum final evidence chunks.
MAX_FINAL_RESULTS = 3


# ============================================================
# DISPUTE CONFIGURATION
# ============================================================

DISPUTE_KEYWORDS: Dict[str, List[str]] = {

    "ITEM_NOT_RECEIVED": [
        "item not received",
        "non delivery",
        "delivery",
        "carrier",
        "tracking",
        "shipment",
        "proof of delivery",
        "delivery confirmation",
        "delivery timestamp",
        "shipping address",
    ],

    "PRODUCT_DEFECTIVE_OR_SWAPPED": [
        "product defect",
        "defective",
        "damaged",
        "swapped",
        "returned product",
        "inspection",
        "fulfillment record",
        "missing components",
        "condition",
    ],

    "UNAUTHORIZED_TRANSACTION": [
        "unauthorized",
        "authentication",
        "account",
        "device",
        "payment",
        "transaction",
        "credentials",
        "transaction metadata",
    ],

    "SUBSCRIPTION_CANCELLED_REFUND": [
        "subscription",
        "cancelled",
        "cancellation",
        "refund",
        "billing",
        "payment record",
        "refund record",
        "cancellation timestamp",
        "subscription status",
    ],

    "NOT_AS_DESCRIBED": [
        "not as described",
        "product description",
        "listing",
        "sku",
        "specifications",
        "order details",
        "fulfillment",
        "product information",
        "photographs",
    ],
}


# Strong policy-section associations.

DISPUTE_SECTION_HINTS: Dict[str, List[str]] = {

    "ITEM_NOT_RECEIVED": [
        "item not received",
        "delivery confirmation",
        "delivery exceptions",
        "shipping",
        "delivery",
    ],

    "PRODUCT_DEFECTIVE_OR_SWAPPED": [
        "product defect",
        "damaged or defective",
        "product swapped",
        "returned in different condition",
        "inspection",
    ],

    "UNAUTHORIZED_TRANSACTION": [
        "unauthorized transactions",
        "account security",
        "authentication",
        "device",
        "transaction records",
    ],

    "SUBSCRIPTION_CANCELLED_REFUND": [
        "subscription cancellation",
        "subscription",
        "refund",
        "refund record",
        "cancellation",
    ],

    "NOT_AS_DESCRIBED": [
        "product not as described",
        "product information",
        "product description",
        "sku",
        "listing",
        "specifications",
    ],
}


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

    documents: List[Dict] = []

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
) -> List[Dict]:
    """
    Section-aware policy chunking.

    Returns structured chunks containing:
        section_index
        section_title
        text
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks: List[Dict] = []

    current_section: List[str] = []
    current_length = 0

    section_index = 0
    section_title = "GENERAL"

    def flush_section():

        nonlocal current_section
        nonlocal current_length
        nonlocal section_index
        nonlocal section_title

        if not current_section:
            return

        section_text = "\n".join(
            current_section
        ).strip()

        chunks.append(
            {
                "section_index": section_index,
                "section_title": section_title,
                "text": section_text,
            }
        )

        section_index += 1

        # Preserve textual overlap only when the section
        # was split because it exceeded chunk_size.
        current_section = []
        current_length = 0

    for line in lines:

        # Detect headings such as:
        #
        # 1. GENERAL REFUND ELIGIBILITY
        # 2. ITEM NOT RECEIVED
        #
        is_section_heading = (
            len(line) > 2
            and line[0].isdigit()
            and "." in line[:4]
        )

        if is_section_heading:

            # Finish previous section.
            flush_section()

            section_title = line

            current_section = [
                line
            ]

            current_length = len(line)

            continue

        current_section.append(line)

        current_length += len(line)

        # Safety split for very large sections.
        if current_length >= chunk_size:

            section_text = "\n".join(
                current_section
            ).strip()

            chunks.append(
                {
                    "section_index": section_index,
                    "section_title": section_title,
                    "text": section_text,
                }
            )

            section_index += 1

            overlap_text = section_text[-overlap:]

            current_section = [
                overlap_text
            ]

            current_length = len(
                overlap_text
            )

    flush_section()

    return chunks


def build_chunks(
    documents: List[Dict],
) -> List[Dict]:
    """
    Convert policy documents into searchable chunks.
    """

    chunks: List[Dict] = []

    for document in documents:

        text_chunks = chunk_text(
            document["text"]
        )

        for chunk in text_chunks:

            chunks.append(
                {
                    "chunk_id": len(chunks),

                    "source":
                        document["source"],

                    "section_index":
                        chunk["section_index"],

                    "section_title":
                        chunk["section_title"],

                    "text":
                        chunk["text"],
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

    print(
        "✅ Embedding model loaded."
    )

    return model


# ============================================================
# VECTOR INDEX
# ============================================================

def build_faiss_index(
    chunks: List[Dict],
    embedding_model,
):
    """
    Generate normalized embeddings and create
    a cosine-similarity FAISS index.
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

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"✅ FAISS index created "
        f"({index.ntotal} vectors)."
    )

    return index


# ============================================================
# SAVE / LOAD
# ============================================================

def save_index(
    index,
    chunks: List[Dict],
):
    """
    Save FAISS index and metadata.
    """

    RAG_STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    np.save(
        METADATA_PATH,
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
        f"   {METADATA_PATH}"
    )


def load_index():
    """
    Load previously built FAISS index
    and chunk metadata.
    """

    if not INDEX_PATH.exists():

        raise FileNotFoundError(
            "FAISS index does not exist. "
            "Run build_rag_index() first."
        )

    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            "Chunk metadata does not exist. "
            "Run build_rag_index() first."
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    chunks = np.load(
        METADATA_PATH,
        allow_pickle=True,
    ).tolist()

    return index, chunks


# ============================================================
# DISPUTE QUERY CONSTRUCTION
# ============================================================

def build_dispute_query(
    query: str,
    dispute_type: Optional[str] = None,
) -> str:
    """
    Enrich the semantic query with dispute-specific
    terminology.

    This improves retrieval without requiring an LLM.
    """

    query = query.strip()

    if not dispute_type:
        return query

    keywords = DISPUTE_KEYWORDS.get(
        dispute_type,
        [],
    )

    keyword_text = " ".join(
        keywords[:8]
    )

    return (
        f"{query}. "
        f"Dispute type: {dispute_type}. "
        f"Relevant policy concepts: "
        f"{keyword_text}"
    )


# ============================================================
# DISPUTE-AWARE RERANKING
# ============================================================

def calculate_keyword_score(
    text: str,
    dispute_type: Optional[str],
) -> float:
    """
    Calculate normalized keyword overlap between
    the policy chunk and dispute-specific concepts.
    """

    if not dispute_type:
        return 0.0

    keywords = DISPUTE_KEYWORDS.get(
        dispute_type,
        [],
    )

    if not keywords:
        return 0.0

    normalized_text = text.lower()

    matches = 0

    for keyword in keywords:

        if keyword.lower() in normalized_text:
            matches += 1

    return min(
        matches / max(len(keywords), 1),
        1.0,
    )


def calculate_section_hint_score(
    text: str,
    dispute_type: Optional[str],
) -> float:
    """
    Score whether the chunk appears to belong to
    the expected dispute-specific policy section.
    """

    if not dispute_type:
        return 0.0

    hints = DISPUTE_SECTION_HINTS.get(
        dispute_type,
        [],
    )

    if not hints:
        return 0.0

    normalized_text = text.lower()

    matches = sum(
        1
        for hint in hints
        if hint.lower() in normalized_text
    )

    return min(
        matches / max(len(hints), 1),
        1.0,
    )


def rerank_result(
    result: Dict,
    dispute_type: Optional[str],
) -> Dict:
    """
    Combine semantic similarity with dispute-specific
    lexical and section relevance.

    Formula:

        reranked =
            0.65 * semantic
          + 0.25 * keyword
          + 0.10 * section_hint
    """

    semantic_score = float(
        result.get(
            "semantic_score",
            result.get("score", 0.0),
        )
    )

    keyword_score = calculate_keyword_score(
        result["text"],
        dispute_type,
    )

    section_score = calculate_section_hint_score(
        result["text"],
        dispute_type,
    )

    reranked_score = (
        0.65 * semantic_score
        + 0.25 * keyword_score
        + 0.10 * section_score
    )

    result["semantic_score"] = round(
        semantic_score,
        4,
    )

    result["keyword_score"] = round(
        keyword_score,
        4,
    )

    result["section_score"] = round(
        section_score,
        4,
    )

    result["reranked_score"] = round(
        reranked_score,
        4,
    )

    return result


# ============================================================
# RELEVANCE LABEL
# ============================================================

def get_relevance_label(
    score: float,
) -> str:
    """
    Convert reranked score into a human-readable label.
    """

    if score >= HIGH_RELEVANCE_THRESHOLD:
        return "HIGH"

    if score >= MEDIUM_RELEVANCE_THRESHOLD:
        return "MEDIUM"

    if score >= MIN_RERANK_SCORE:
        return "LOW"

    return "FILTERED"


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_results(
    results: List[Dict],
) -> List[Dict]:
    """
    Remove duplicate policy chunks.

    Deduplication is based on normalized text.
    """

    seen = set()

    unique_results = []

    for result in results:

        normalized = " ".join(
            result["text"]
            .lower()
            .split()
        )

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        unique_results.append(
            result
        )

    return unique_results


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(
    query: str,
    embedding_model,
    index,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
    dispute_type: Optional[str] = None,
    min_relevance: float = MIN_RERANK_SCORE,
) -> List[Dict]:
    """
    Retrieve, rerank, filter and deduplicate policy evidence.

    Returns only policy chunks that pass the final
    relevance threshold.

    Each result contains:

        source
        section_index
        section_title
        text
        semantic_score
        keyword_score
        section_score
        reranked_score
        relevance_label
        retrieval_rank
    """

    enriched_query = build_dispute_query(
        query,
        dispute_type,
    )

    query_embedding = embedding_model.encode(
        [enriched_query],
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    faiss.normalize_L2(
        query_embedding
    )

    candidate_k = min(
        max(
            top_k,
            RETRIEVAL_CANDIDATES,
        ),
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        candidate_k,
    )

    candidates: List[Dict] = []

    for semantic_score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position == -1:
            continue

        chunk = chunks[
            index_position
        ]

        result = {
            "source":
                chunk["source"],

            "section_index":
                chunk["section_index"],

            "section_title":
                chunk.get(
                    "section_title",
                    "",
                ),

            "text":
                chunk["text"],

            "semantic_score":
                float(semantic_score),

            "retrieval_rank":
                len(candidates) + 1,
        }

        result = rerank_result(
            result,
            dispute_type,
        )

        result["relevance_label"] = (
            get_relevance_label(
                result["reranked_score"]
            )
        )

        candidates.append(
            result
        )

    # Highest reranked result first.
    candidates.sort(
        key=lambda item: item[
            "reranked_score"
        ],
        reverse=True,
    )

    # Remove irrelevant chunks.
    filtered = [
        result
        for result in candidates
        if result["reranked_score"]
        >= min_relevance
    ]

    # Deduplicate.
    filtered = deduplicate_results(
        filtered
    )

    # Return only requested final evidence count.
    final_results = filtered[
        :min(
            top_k,
            MAX_FINAL_RESULTS,
        )
    ]

    return final_results


# ============================================================
# RETRIEVAL DIAGNOSTICS
# ============================================================

def retrieve_with_diagnostics(
    query: str,
    embedding_model,
    index,
    chunks: List[Dict],
    top_k: int = DEFAULT_TOP_K,
    dispute_type: Optional[str] = None,
    min_relevance: float = MIN_RERANK_SCORE,
) -> Tuple[List[Dict], Dict]:
    """
    Retrieval wrapper returning both evidence and
    diagnostic statistics.

    Useful for the Streamlit RAG dashboard.
    """

    enriched_query = build_dispute_query(
        query,
        dispute_type,
    )

    query_embedding = embedding_model.encode(
        [enriched_query],
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    faiss.normalize_L2(
        query_embedding
    )

    candidate_k = min(
        max(
            top_k,
            RETRIEVAL_CANDIDATES,
        ),
        index.ntotal,
    )

    scores, indices = index.search(
        query_embedding,
        candidate_k,
    )

    candidates: List[Dict] = []

    for semantic_score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position == -1:
            continue

        chunk = chunks[
            index_position
        ]

        result = {
            "source":
                chunk["source"],

            "section_index":
                chunk["section_index"],

            "section_title":
                chunk.get(
                    "section_title",
                    "",
                ),

            "text":
                chunk["text"],

            "semantic_score":
                float(semantic_score),

            "retrieval_rank":
                len(candidates) + 1,
        }

        result = rerank_result(
            result,
            dispute_type,
        )

        result["relevance_label"] = (
            get_relevance_label(
                result["reranked_score"]
            )
        )

        candidates.append(
            result
        )

    candidates.sort(
        key=lambda item: item[
            "reranked_score"
        ],
        reverse=True,
    )

    before_filtering = len(
        candidates
    )

    filtered = [
        result
        for result in candidates
        if result["reranked_score"]
        >= min_relevance
    ]

    after_threshold = len(
        filtered
    )

    deduplicated = deduplicate_results(
        filtered
    )

    after_deduplication = len(
        deduplicated
    )

    final_results = deduplicated[
        :min(
            top_k,
            MAX_FINAL_RESULTS,
        )
    ]

    diagnostics = {

        "query":
            query,

        "enriched_query":
            enriched_query,

        "dispute_type":
            dispute_type,

        "candidate_count":
            before_filtering,

        "passed_threshold":
            after_threshold,

        "duplicates_removed":
            after_threshold
            - after_deduplication,

        "filtered_out":
            before_filtering
            - after_threshold,

        "final_result_count":
            len(final_results),

        "min_relevance":
            min_relevance,

        "top_score":
            (
                final_results[0][
                    "reranked_score"
                ]
                if final_results
                else 0.0
            ),

        "top_source":
            (
                final_results[0][
                    "source"
                ]
                if final_results
                else None
            ),

        "top_section":
            (
                final_results[0][
                    "section_title"
                ]
                if final_results
                else None
            ),
    }

    return (
        final_results,
        diagnostics,
    )


# ============================================================
# BUILD COMPLETE RAG INDEX
# ============================================================

def build_rag_index():
    """
    Complete indexing pipeline.
    """

    print("\n" + "=" * 70)
    print(
        "🔎 HORAЕ RAG INDEX BUILD"
    )
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
    print(
        "🎉 RAG INDEX READY"
    )
    print("=" * 70)

    return (
        embedding_model,
        index,
        chunks,
    )


# ============================================================
# EVALUATION DATASET
# ============================================================

RAG_EVALUATION_CASES = [

    {
        "query":
            "Customer claims that their package was not received.",

        "dispute_type":
            "ITEM_NOT_RECEIVED",
    },

    {
        "query":
            "Customer says the product was defective.",

        "dispute_type":
            "PRODUCT_DEFECTIVE_OR_SWAPPED",
    },

    {
        "query":
            "Customer claims the transaction was unauthorized.",

        "dispute_type":
            "UNAUTHORIZED_TRANSACTION",
    },

    {
        "query":
            "Customer claims they cancelled the subscription but did not receive a refund.",

        "dispute_type":
            "SUBSCRIPTION_CANCELLED_REFUND",
    },

    {
        "query":
            "Customer says the product was not as described.",

        "dispute_type":
            "NOT_AS_DESCRIBED",
    },
]


# ============================================================
# EXPECTED POLICY TERMS
# ============================================================

EXPECTED_POLICY_TERMS = {

    "ITEM_NOT_RECEIVED":
        [
            "ITEM NOT RECEIVED",
        ],

    "PRODUCT_DEFECTIVE_OR_SWAPPED":
        [
            "PRODUCT DEFECT",
        ],

    "UNAUTHORIZED_TRANSACTION":
        [
            "UNAUTHORIZED TRANSACTIONS",
        ],

    "SUBSCRIPTION_CANCELLED_REFUND":
        [
            "SUBSCRIPTION CANCELLATION",
        ],

    "NOT_AS_DESCRIBED":
        [
            "PRODUCT NOT AS DESCRIBED",
        ],
}


def evaluate_retrieval(
    embedding_model,
    index,
    chunks: List[Dict],
) -> Dict:
    """
    Evaluate whether each supported dispute type retrieves
    its expected policy section at rank #1.

    This is a lightweight regression test rather than a
    statistical benchmark.
    """

    print("\n" + "=" * 70)
    print(
        "📊 HORAЕ RAG RELEVANCE EVALUATION"
    )
    print("=" * 70)

    results = []

    passed = 0

    for case in RAG_EVALUATION_CASES:

        query = case["query"]
        dispute_type = case[
            "dispute_type"
        ]

        retrieved, diagnostics = (
            retrieve_with_diagnostics(
                query=query,
                embedding_model=embedding_model,
                index=index,
                chunks=chunks,
                top_k=3,
                dispute_type=dispute_type,
            )
        )

        expected_terms = (
            EXPECTED_POLICY_TERMS.get(
                dispute_type,
                [],
            )
        )

        top_match = (
            retrieved[0]
            if retrieved
            else None
        )

        top_text = (
            top_match["text"].upper()
            if top_match
            else ""
        )

        expected_found = any(
            term.upper()
            in top_text
            for term in expected_terms
        )

        if expected_found:
            passed += 1

        results.append(
            {
                "dispute_type":
                    dispute_type,

                "passed":
                    expected_found,

                "top_source":
                    diagnostics[
                        "top_source"
                    ],

                "top_section":
                    diagnostics[
                        "top_section"
                    ],

                "top_score":
                    diagnostics[
                        "top_score"
                    ],

                "filtered_out":
                    diagnostics[
                        "filtered_out"
                    ],
            }
        )

        status = (
            "✅ PASS"
            if expected_found
            else "❌ FAIL"
        )

        print(
            f"\n{status} | {dispute_type}"
        )

        print(
            f"Top section: "
            f"{diagnostics['top_section']}"
        )

        print(
            f"Reranked score: "
            f"{diagnostics['top_score']:.3f}"
        )

        print(
            f"Filtered: "
            f"{diagnostics['filtered_out']}"
        )

    accuracy = (
        passed / len(
            RAG_EVALUATION_CASES
        )
        if RAG_EVALUATION_CASES
        else 0.0
    )

    summary = {
        "cases":
            len(RAG_EVALUATION_CASES),

        "passed":
            passed,

        "failed":
            len(
                RAG_EVALUATION_CASES
            ) - passed,

        "retrieval_accuracy":
            accuracy,

        "results":
            results,
    }

    print("\n" + "-" * 70)

    print(
        f"RAG Retrieval Accuracy: "
        f"{passed}/{len(RAG_EVALUATION_CASES)} "
        f"({accuracy * 100:.1f}%)"
    )

    print("=" * 70)

    return summary


# ============================================================
# TEST RETRIEVAL
# ============================================================

def test_retrieval():
    """
    Run the full RAG retrieval regression suite.
    """

    embedding_model, index, chunks = (
        build_rag_index()
    )

    for case in RAG_EVALUATION_CASES:

        query = case["query"]
        dispute_type = case[
            "dispute_type"
        ]

        print("\n" + "-" * 70)

        print(
            f"QUERY: {query}"
        )

        print(
            f"DISPUTE: {dispute_type}"
        )

        print("-" * 70)

        results, diagnostics = (
            retrieve_with_diagnostics(
                query=query,
                embedding_model=embedding_model,
                index=index,
                chunks=chunks,
                top_k=3,
                dispute_type=dispute_type,
            )
        )

        if not results:

            print(
                "⚠️ No evidence passed the relevance threshold."
            )

            continue

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\n[{rank}] "
                f"{result['source']} "
                f"| Section: "
                f"{result['section_index']} "
                f"| Relevance: "
                f"{result['relevance_label']}"
            )

            print(
                f"Semantic: "
                f"{result['semantic_score']:.3f} "
                f"| Keyword: "
                f"{result['keyword_score']:.3f} "
                f"| Section: "
                f"{result['section_score']:.3f} "
                f"| Reranked: "
                f"{result['reranked_score']:.3f}"
            )

            print(
                result["text"]
            )

        print(
            "\n📊 Retrieval diagnostics:"
        )

        print(
            f"Candidates: "
            f"{diagnostics['candidate_count']}"
        )

        print(
            f"Filtered out: "
            f"{diagnostics['filtered_out']}"
        )

        print(
            f"Final evidence: "
            f"{diagnostics['final_result_count']}"
        )

    evaluate_retrieval(
        embedding_model,
        index,
        chunks,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_retrieval()