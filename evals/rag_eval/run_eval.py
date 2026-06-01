"""RAG Evaluation Script — compares rag_enhanced vs original ContextCompressor.

Metrics:
    1. Retrieval Recall: What fraction of expected keywords are found in retrieved context?
    2. Retrieval Precision: How many retrieved chunks contain relevant keywords?
    3. Context Quality: LLM-based score (0-10) of how well context answers the question.

Usage:
    python -m evals.rag_eval.run_eval

    # Quick test without LLM (only retrieval metrics):
    SKIP_LLM_JUDGE=1 python -m evals.rag_eval.run_eval
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from evals.rag_eval.eval_dataset import EVAL_DATASET

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_keyword_recall(context: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the retrieved context."""
    if not expected_keywords:
        return 1.0
    context_lower = context.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
    return found / len(expected_keywords)


def compute_keyword_precision(context: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords that are relevant (non-zero recall = precision hit)."""
    if not expected_keywords or not context:
        return 0.0
    context_lower = context.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
    return hits / len(expected_keywords)


def compute_hit_rate(context: str, expected_keywords: list[str]) -> bool:
    """Whether at least one expected keyword is found."""
    context_lower = context.lower()
    return any(kw.lower() in context_lower for kw in expected_keywords)


async def llm_judge_score(question: str, context: str, ground_truth: str) -> float:
    """Use LLM to rate context quality on a 0-10 scale."""
    if os.environ.get("SKIP_LLM_JUDGE"):
        return -1.0  # skipped

    from gpt_researcher.utils.llm import create_chat_completion
    from gpt_researcher.config import Config

    cfg = Config()
    prompt = f"""Rate how well the following retrieved context answers the question on a scale of 0-10.
Consider both relevance and completeness compared to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Context:
{context[:3000]}

Rating (0-10, reply with ONLY the number):"""

    try:
        response = await create_chat_completion(
            model=cfg.fast_llm_model,
            messages=[{"role": "user", "content": prompt}],
            llm_provider=cfg.fast_llm_provider,
        )
        # Extract number from response
        score_str = response.strip().split()[0]
        return float(score_str)
    except Exception as e:
        print(f"  LLM judge error: {e}")
        return -1.0


# ---------------------------------------------------------------------------
# System under test
# ---------------------------------------------------------------------------


async def run_original_system(query: str, documents: list[dict]) -> str:
    """Run the original ContextCompressor."""
    from gpt_researcher.context.compression import ContextCompressor
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)
    compressor = ContextCompressor(
        documents=documents,
        embeddings=memory.get_embeddings(),
    )
    return await compressor.async_get_context(query=query, max_results=10)


async def run_enhanced_system(query: str, documents: list[dict], store=None) -> str:
    """Run the enhanced rag_enhanced pipeline."""
    from rag_enhanced.config import RAGConfig
    from rag_enhanced.pipeline import RAGPipeline
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)

    config = RAGConfig(
        enable_query_rewrite=False,  # keep it fair
        hybrid_search=True,
        rerank_top_k=10,
        max_results=10,
    )

    pipeline = RAGPipeline(
        config=config,
        embeddings=memory.get_embeddings(),
        knowledge_store=store,
    )
    return await pipeline.process(query, documents)


async def run_enhanced_with_kb(query: str, store) -> str:
    """Run enhanced pipeline in primary mode (KB only)."""
    from rag_enhanced.config import RAGConfig
    from rag_enhanced.pipeline import RAGPipeline
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)

    config = RAGConfig(
        knowledge_store_mode="primary",
        enable_query_rewrite=False,
        rerank_top_k=10,
        max_results=10,
    )

    pipeline = RAGPipeline(
        config=config,
        embeddings=memory.get_embeddings(),
        knowledge_store=store,
    )
    return await pipeline.process(query, [])


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


async def evaluate():
    """Run evaluation on all test cases."""
    print("=" * 70)
    print("RAG Evaluation: rag_enhanced vs Original ContextCompressor")
    print("=" * 70)

    # Load test document
    doc_path = Path(__file__).resolve().parents[2] / "my-docs" / "sample.txt"
    if not doc_path.exists():
        print(f"ERROR: Test document not found at {doc_path}")
        return

    with open(doc_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    documents = [{"raw_content": raw_content, "url": "sample.txt", "title": "GPT Researcher 技术文档"}]

    # Build knowledge store for KB-only test
    print("\n[*] Building knowledge store from test document...")
    from rag_enhanced.knowledge_store.chroma_store import ChromaKnowledgeStore
    from rag_enhanced.chunking.adaptive import AdaptiveChunker
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)
    embeddings = memory.get_embeddings()

    tmpdir = tempfile.mkdtemp()
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="eval_test",
        persist_directory=tmpdir,
    )
    chunker = AdaptiveChunker()
    chunks = await chunker.chunk(documents)
    await store.add_documents(chunks)
    print(f"   Ingested {len(chunks)} chunks into knowledge store")

    # Results storage
    results = {
        "original": {"recall": [], "precision": [], "hit_rate": [], "llm_score": [], "time": []},
        "enhanced": {"recall": [], "precision": [], "hit_rate": [], "llm_score": [], "time": []},
        "kb_only":  {"recall": [], "precision": [], "hit_rate": [], "llm_score": [], "time": []},
    }

    for item in EVAL_DATASET:
        qid = item["id"]
        question = item["question"]
        gt = item["ground_truth"]
        keywords = item["expected_keywords"]

        print(f"\n--- Q{qid}: {question[:50]}... ---")

        # --- Original system ---
        t0 = time.perf_counter()
        try:
            ctx_orig = await run_original_system(question, documents)
            t_orig = time.perf_counter() - t0
        except Exception as e:
            print(f"  Original error: {e}")
            ctx_orig = ""
            t_orig = 0

        recall_orig = compute_keyword_recall(ctx_orig, keywords)
        hit_orig = compute_hit_rate(ctx_orig, keywords)

        results["original"]["recall"].append(recall_orig)
        results["original"]["hit_rate"].append(hit_orig)
        results["original"]["time"].append(t_orig)

        # --- Enhanced system (in-memory) ---
        t0 = time.perf_counter()
        try:
            ctx_enh = await run_enhanced_system(question, documents)
            t_enh = time.perf_counter() - t0
        except Exception as e:
            print(f"  Enhanced error: {e}")
            ctx_enh = ""
            t_enh = 0

        recall_enh = compute_keyword_recall(ctx_enh, keywords)
        hit_enh = compute_hit_rate(ctx_enh, keywords)

        results["enhanced"]["recall"].append(recall_enh)
        results["enhanced"]["hit_rate"].append(hit_enh)
        results["enhanced"]["time"].append(t_enh)

        # --- Enhanced system (KB only) ---
        t0 = time.perf_counter()
        try:
            ctx_kb = await run_enhanced_with_kb(question, store)
            t_kb = time.perf_counter() - t0
        except Exception as e:
            print(f"  KB-only error: {e}")
            ctx_kb = ""
            t_kb = 0

        recall_kb = compute_keyword_recall(ctx_kb, keywords)
        hit_kb = compute_hit_rate(ctx_kb, keywords)

        results["kb_only"]["recall"].append(recall_kb)
        results["kb_only"]["hit_rate"].append(hit_kb)
        results["kb_only"]["time"].append(t_kb)

        # Print per-question results
        print(f"  Original:  recall={recall_orig:.0%}  hit={hit_orig}  time={t_orig:.3f}s  ctx_len={len(ctx_orig)}")
        print(f"  Enhanced:  recall={recall_enh:.0%}  hit={hit_enh}  time={t_enh:.3f}s  ctx_len={len(ctx_enh)}")
        print(f"  KB-only:   recall={recall_kb:.0%}  hit={hit_kb}  time={t_kb:.3f}s  ctx_len={len(ctx_kb)}")

    # --- LLM Judge (optional) ---
    if not os.environ.get("SKIP_LLM_JUDGE"):
        print("\n[*] Running LLM judge scoring...")
        for item in EVAL_DATASET:
            qid = item["id"]
            question = item["question"]
            gt = item["ground_truth"]

            # Re-run to get contexts (they're not stored from above)
            try:
                ctx_orig = await run_original_system(question, documents)
            except:
                ctx_orig = ""
            try:
                ctx_enh = await run_enhanced_system(question, documents)
            except:
                ctx_enh = ""

            score_orig = await llm_judge_score(question, ctx_orig, gt)
            score_enh = await llm_judge_score(question, ctx_enh, gt)

            results["original"]["llm_score"].append(score_orig)
            results["enhanced"]["llm_score"].append(score_enh)

            print(f"  Q{qid}: Original={score_orig:.1f}  Enhanced={score_enh:.1f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for system_name, data in results.items():
        avg_recall = sum(data["recall"]) / len(data["recall"]) if data["recall"] else 0
        avg_hit = sum(data["hit_rate"]) / len(data["hit_rate"]) if data["hit_rate"] else 0
        avg_time = sum(data["time"]) / len(data["time"]) if data["time"] else 0
        avg_llm = sum(data["llm_score"]) / len(data["llm_score"]) if data["llm_score"] else -1

        print(f"\n  {system_name.upper()}:")
        print(f"    Keyword Recall:  {avg_recall:.1%}")
        print(f"    Hit Rate:        {avg_hit:.1%}  (at least one keyword found)")
        print(f"    Avg Latency:     {avg_time:.3f}s")
        if avg_llm >= 0:
            print(f"    LLM Judge Score: {avg_llm:.1f}/10")

    # Save results to JSON
    output_path = Path(__file__).resolve().parents[2] / "evals" / "rag_eval" / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_size": len(EVAL_DATASET),
            "per_question": [
                {
                    "id": item["id"],
                    "question": item["question"],
                    "original_recall": results["original"]["recall"][i],
                    "enhanced_recall": results["enhanced"]["recall"][i],
                    "kb_only_recall": results["kb_only"]["recall"][i],
                    "original_hit": results["original"]["hit_rate"][i],
                    "enhanced_hit": results["enhanced"]["hit_rate"][i],
                    "kb_only_hit": results["kb_only"]["hit_rate"][i],
                }
                for i, item in enumerate(EVAL_DATASET)
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Results saved to {output_path}")

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except:
        pass


if __name__ == "__main__":
    asyncio.run(evaluate())
