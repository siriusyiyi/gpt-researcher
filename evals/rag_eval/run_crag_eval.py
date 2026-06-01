"""CRAG Benchmark Evaluation for rag_enhanced module.

Evaluates retrieval quality using Facebook Research's CRAG dataset
(Comprehensive RAG Benchmark) across three systems:
  1. Original ContextCompressor (embedding similarity filter)
  2. Enhanced Pipeline (hybrid BM25+vector, reranking, compression)
  3. KB-only (KnowledgeStore primary mode)

Usage:
    python -m evals.rag_eval.run_crag_eval
    python -m evals.rag_eval.run_crag_eval --sample 30 --domains sports,movie
    python -m evals.rag_eval.run_crag_eval --no-judge
    python -m evals.rag_eval.run_crag_eval --max-pages 100000

Environment:
    OPENAI_API_KEY + OPENAI_BASE_URL  — for embeddings + default LLM
    DEEPSEEK_API_KEY                  — (optional) separate DeepSeek API for judge
    JUDGE_LLM                         — (optional) "provider:model" for judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

def strip_html(html: str, max_chars: int = 50_000) -> str:
    """Strip HTML tags, return plain text truncated to max_chars."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        # Parse only the first portion to save memory on huge pages
        soup = BeautifulSoup(html[:max_chars * 2], "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception:
        # Fallback: crude tag removal
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        return " ".join(text.split())[:max_chars]


# ---------------------------------------------------------------------------
# CRAG data loading
# ---------------------------------------------------------------------------

CRAG_DATA_PATH = Path(__file__).parent / "crag_data" / "crag_task1_dev.jsonl.bz2"


def load_crag_data(split: int = 0) -> list[dict]:
    """Load CRAG Task 1 dev data, filtered by split."""
    import bz2

    records = []
    with bz2.open(CRAG_DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == split:
                records.append(r)
    return records


def sample_questions(
    data: list[dict],
    n_per_domain: int = 10,
    domains: list[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Stratified sampling: n_per_domain questions per domain."""
    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = {}
    for r in data:
        d = r["domain"]
        if domains and d not in domains:
            continue
        by_domain.setdefault(d, []).append(r)

    sampled = []
    for d, items in sorted(by_domain.items()):
        rng.shuffle(items)
        sampled.extend(items[:n_per_domain])
        print(f"  {d}: sampled {min(n_per_domain, len(items))}/{len(items)}")

    rng.shuffle(sampled)
    return sampled


def extract_documents(
    record: dict,
    max_page_chars: int = 50_000,
) -> list[dict]:
    """Extract plain-text documents from CRAG search_results."""
    docs = []
    for i, sr in enumerate(record.get("search_results", [])):
        html = sr.get("page_result", "")
        text = strip_html(html, max_chars=max_page_chars)
        if not text.strip():
            # Fall back to snippet if page_result is empty
            text = sr.get("page_snippet", "")
        url = sr.get("page_url", f"page_{i}")
        title = sr.get("page_name", f"Page {i}")
        docs.append({
            "raw_content": text,
            "url": url,
            "title": title,
        })
    return docs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def token_recall(context: str, answer: str) -> float:
    """Fraction of gold answer tokens found in context (case-insensitive)."""
    if not answer:
        return 1.0
    ctx_lower = context.lower()
    answer_tokens = answer.lower().split()
    if not answer_tokens:
        return 1.0
    found = sum(1 for t in answer_tokens if t in ctx_lower)
    return found / len(answer_tokens)


def key_phrase_recall(context: str, answer: str) -> float:
    """Fraction of meaningful phrases (2-grams+) from answer found in context."""
    if not answer:
        return 1.0
    ctx_lower = context.lower()
    # Extract key phrases: remove common stop words and check
    answer_lower = answer.lower()
    # Simple approach: split on punctuation, check each phrase
    import re
    phrases = re.split(r'[,.:;!?()]', answer_lower)
    phrases = [p.strip() for p in phrases if len(p.strip()) > 5]

    if not phrases:
        return 1.0

    found = sum(1 for p in phrases if p in ctx_lower)
    return found / len(phrases)


# ---------------------------------------------------------------------------
# Systems under test
# ---------------------------------------------------------------------------

async def run_original(query: str, documents: list[dict]) -> tuple[str, float]:
    """Original ContextCompressor. Returns (context, latency)."""
    from gpt_researcher.context.compression import ContextCompressor
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)
    compressor = ContextCompressor(
        documents=documents,
        embeddings=memory.get_embeddings(),
    )
    t0 = time.perf_counter()
    ctx = await compressor.async_get_context(query=query, max_results=10)
    latency = time.perf_counter() - t0
    return ctx, latency


async def run_enhanced(
    query: str,
    documents: list[dict],
    store=None,
) -> tuple[str, float]:
    """Enhanced RAG pipeline (supplement mode). Returns (context, latency)."""
    from rag_enhanced.config import RAGConfig
    from rag_enhanced.pipeline import RAGPipeline
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)

    config = RAGConfig(
        enable_query_rewrite=False,
        hybrid_search=True,
        rerank_top_k=10,
        max_results=10,
    )

    pipeline = RAGPipeline(
        config=config,
        embeddings=memory.get_embeddings(),
        knowledge_store=store,
    )
    t0 = time.perf_counter()
    ctx = await pipeline.process(query, documents)
    latency = time.perf_counter() - t0
    return ctx, latency


async def run_kb_only(query: str, store) -> tuple[str, float]:
    """KB-only (primary mode). Returns (context, latency)."""
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
    t0 = time.perf_counter()
    ctx = await pipeline.process(query, [])
    latency = time.perf_counter() - t0
    return ctx, latency


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

async def llm_judge(
    question: str,
    context: str,
    gold_answer: str,
    model: str | None = None,
    provider: str | None = None,
) -> float:
    """LLM judges context quality on 0-10 scale.

    Uses project's LLM by default, or DeepSeek if DEEPSEEK_API_KEY is set.
    """
    if model is None or provider is None:
        # Use project default
        from gpt_researcher.config import Config
        cfg = Config()
        model = model or cfg.fast_llm_model
        provider = provider or cfg.fast_llm_provider

    from gpt_researcher.utils.llm import create_chat_completion

    prompt = f"""Rate how well the retrieved CONTEXT contains the information needed to answer the QUESTION, compared to the GOLD ANSWER.

Scale:
  10 = Context contains all key information from the gold answer
  7-9 = Context contains most key information, minor gaps
  4-6 = Context contains some relevant information but missing key parts
  1-3 = Context has little relevant information
  0 = Context is completely irrelevant

QUESTION: {question}

GOLD ANSWER: {gold_answer}

RETRIEVED CONTEXT (first 3000 chars):
{context[:3000]}

Rating (reply with ONLY a number 0-10):"""

    try:
        response = await create_chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            llm_provider=provider,
            temperature=0.0,
            max_tokens=10,
        )
        score_str = response.strip().split()[0]
        return float(score_str)
    except Exception as e:
        print(f"    [!] LLM judge error: {e}")
        return -1.0


def get_judge_config():
    """Resolve LLM judge model and provider."""
    # Check for explicit JUDGE_LLM env var: "provider:model"
    judge_llm = os.environ.get("JUDGE_LLM")
    if judge_llm and ":" in judge_llm:
        provider, model = judge_llm.split(":", 1)
        return provider, model

    # Check for DEEPSEEK_API_KEY → use native DeepSeek provider
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek", "deepseek-chat"

    # Fall back to project default
    from gpt_researcher.config import Config
    cfg = Config()
    return cfg.fast_llm_provider, cfg.fast_llm_model


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def evaluate(
    sample_size: int = 10,
    domains: list[str] | None = None,
    max_page_chars: int = 50_000,
    skip_judge: bool = False,
    seed: int = 42,
):
    """Run CRAG evaluation."""
    print("=" * 72)
    print("CRAG Benchmark Evaluation: rag_enhanced vs Original vs KB-only")
    print("=" * 72)

    # --- Load data ---
    print("\n[*] Loading CRAG data...")
    data = load_crag_data(split=0)
    print(f"    Loaded {len(data)} questions (split=0, validation)")

    # --- Sample ---
    print(f"\n[*] Sampling {sample_size} questions per domain...")
    sampled = sample_questions(data, n_per_domain=sample_size, domains=domains, seed=seed)
    print(f"    Total sampled: {len(sampled)} questions")

    # Domain / type distribution
    domain_counts = Counter(r["domain"] for r in sampled)
    type_counts = Counter(r["question_type"] for r in sampled)
    print(f"    Domains: {dict(domain_counts)}")
    print(f"    Types: {dict(type_counts)}")

    # --- Prepare KnowledgeStore for KB-only tests ---
    print("\n[*] Preparing KnowledgeStore...")
    from rag_enhanced.knowledge_store.chroma_store import ChromaKnowledgeStore
    from rag_enhanced.chunking.adaptive import AdaptiveChunker
    from gpt_researcher.memory import Memory
    from gpt_researcher.config import Config

    cfg = Config()
    memory = Memory(cfg.embedding_provider, cfg.embedding_model)
    embeddings = memory.get_embeddings()

    tmpdir = tempfile.mkdtemp(prefix="crag_eval_")
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="crag_eval",
        persist_directory=tmpdir,
    )
    chunker = AdaptiveChunker()

    # --- Results ---
    results = {
        "original":   {"token_recall": [], "phrase_recall": [], "judge": [], "time": []},
        "enhanced":   {"token_recall": [], "phrase_recall": [], "judge": [], "time": []},
        "kb_only":    {"token_recall": [], "phrase_recall": [], "judge": [], "time": []},
    }
    per_question = []

    # --- Evaluate each question ---
    for qi, record in enumerate(sampled, 1):
        qid = record["interaction_id"][:8]
        question = record["query"]
        gold = record["answer"]
        domain = record["domain"]
        qtype = record["question_type"]

        print(f"\n--- [{qi}/{len(sampled)}] ({domain}/{qtype}) {question[:70]}... ---")

        docs = extract_documents(record, max_page_chars=max_page_chars)
        total_chars = sum(len(d["raw_content"]) for d in docs)
        print(f"    Docs: {len(docs)} pages, {total_chars:,} chars total")

        # --- Original ---
        try:
            ctx_orig, t_orig = await run_original(question, docs)
        except Exception as e:
            print(f"    [!] Original error: {e}")
            ctx_orig, t_orig = "", 0

        tr_orig = token_recall(ctx_orig, gold)
        pr_orig = key_phrase_recall(ctx_orig, gold)

        results["original"]["token_recall"].append(tr_orig)
        results["original"]["phrase_recall"].append(pr_orig)
        results["original"]["time"].append(t_orig)

        # --- Enhanced (supplement, in-memory) ---
        try:
            ctx_enh, t_enh = await run_enhanced(question, docs)
        except Exception as e:
            print(f"    [!] Enhanced error: {e}")
            ctx_enh, t_enh = "", 0

        tr_enh = token_recall(ctx_enh, gold)
        pr_enh = key_phrase_recall(ctx_enh, gold)

        results["enhanced"]["token_recall"].append(tr_enh)
        results["enhanced"]["phrase_recall"].append(pr_enh)
        results["enhanced"]["time"].append(t_enh)

        # --- KB-only ---
        # Ingest documents into KB, tag source for cleanup
        kb_source = f"crag_{qi}_{qid}"
        try:
            chunks = await chunker.chunk(docs)
            if chunks:
                # Override source so add_documents dedup works correctly
                for c in chunks:
                    c.metadata["source"] = kb_source
                await store.add_documents(chunks)

            ctx_kb, t_kb = await run_kb_only(question, store)
        except Exception as e:
            print(f"    [!] KB-only error: {e}")
            ctx_kb, t_kb = "", 0
        finally:
            # Cleanup this question's chunks from KB
            try:
                await store.delete(kb_source)
            except Exception:
                pass

        tr_kb = token_recall(ctx_kb, gold)
        pr_kb = key_phrase_recall(ctx_kb, gold)

        results["kb_only"]["token_recall"].append(tr_kb)
        results["kb_only"]["phrase_recall"].append(pr_kb)
        results["kb_only"]["time"].append(t_kb)

        # Per-question print
        print(f"    Original:  TR={tr_orig:.0%} PR={pr_orig:.0%} t={t_orig:.3f}s ctx={len(ctx_orig):,}")
        print(f"    Enhanced:  TR={tr_enh:.0%} PR={pr_enh:.0%} t={t_enh:.3f}s ctx={len(ctx_enh):,}")
        print(f"    KB-only:   TR={tr_kb:.0%} PR={pr_kb:.0%} t={t_kb:.3f}s ctx={len(ctx_kb):,}")

        per_question.append({
            "id": qid,
            "domain": domain,
            "question_type": qtype,
            "question": question,
            "gold_answer": gold,
            "original": {"token_recall": tr_orig, "phrase_recall": pr_orig,
                         "time": t_orig, "ctx_len": len(ctx_orig)},
            "enhanced": {"token_recall": tr_enh, "phrase_recall": pr_enh,
                         "time": t_enh, "ctx_len": len(ctx_enh)},
            "kb_only":  {"token_recall": tr_kb, "phrase_recall": pr_kb,
                         "time": t_kb, "ctx_len": len(ctx_kb)},
        })

    # --- LLM Judge ---
    judge_scores = {"original": [], "enhanced": []}
    if not skip_judge:
        judge_provider, judge_model = get_judge_config()
        print(f"\n[*] LLM Judge: provider={judge_provider}, model={judge_model}")

        for qi, record in enumerate(sampled, 1):
            question = record["query"]
            gold = record["answer"]
            pq = per_question[qi - 1]

            # Re-run to get context (not stored from above to save memory)
            docs = extract_documents(record, max_page_chars=max_page_chars)

            try:
                ctx_orig, _ = await run_original(question, docs)
                score_orig = await llm_judge(question, ctx_orig, gold,
                                             model=judge_model, provider=judge_provider)
            except Exception as e:
                print(f"    [!] Judge original error: {e}")
                score_orig = -1.0

            try:
                ctx_enh, _ = await run_enhanced(question, docs)
                score_enh = await llm_judge(question, ctx_enh, gold,
                                            model=judge_model, provider=judge_provider)
            except Exception as e:
                print(f"    [!] Judge enhanced error: {e}")
                score_enh = -1.0

            judge_scores["original"].append(score_orig)
            judge_scores["enhanced"].append(score_enh)

            pq["original"]["judge"] = score_orig
            pq["enhanced"]["judge"] = score_enh

            print(f"    [{qi}/{len(sampled)}] Orig={score_orig:.1f}  Enh={score_enh:.1f}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for system_name in ["original", "enhanced", "kb_only"]:
        data = results[system_name]
        n = len(data["token_recall"])

        avg_tr = sum(data["token_recall"]) / n if n else 0
        avg_pr = sum(data["phrase_recall"]) / n if n else 0
        avg_time = sum(data["time"]) / n if n else 0

        print(f"\n  {system_name.upper()}:")
        print(f"    Token Recall:     {avg_tr:.1%}")
        print(f"    Phrase Recall:    {avg_pr:.1%}")
        print(f"    Avg Latency:      {avg_time:.3f}s")

        if judge_scores.get(system_name):
            valid = [s for s in judge_scores[system_name] if s >= 0]
            avg_judge = sum(valid) / len(valid) if valid else 0
            print(f"    LLM Judge Score:  {avg_judge:.1f}/10  ({len(valid)} scored)")

    # --- Breakdown by domain ---
    print("\n--- Breakdown by Domain ---")
    for domain in sorted(domain_counts.keys()):
        domain_qs = [pq for pq in per_question if pq["domain"] == domain]
        print(f"\n  {domain.upper()} ({len(domain_qs)} questions):")
        for sys_name in ["original", "enhanced", "kb_only"]:
            trs = [q[sys_name]["token_recall"] for q in domain_qs]
            prs = [q[sys_name]["phrase_recall"] for q in domain_qs]
            avg_tr = sum(trs) / len(trs)
            avg_pr = sum(prs) / len(prs)
            print(f"    {sys_name:12s}: TR={avg_tr:.0%}  PR={avg_pr:.0%}")

    # --- Breakdown by question type ---
    print("\n--- Breakdown by Question Type ---")
    for qtype in sorted(type_counts.keys()):
        type_qs = [pq for pq in per_question if pq["question_type"] == qtype]
        print(f"\n  {qtype} ({len(type_qs)} questions):")
        for sys_name in ["original", "enhanced", "kb_only"]:
            trs = [q[sys_name]["token_recall"] for q in type_qs]
            prs = [q[sys_name]["phrase_recall"] for q in type_qs]
            avg_tr = sum(trs) / len(trs)
            avg_pr = sum(prs) / len(prs)
            print(f"    {sys_name:12s}: TR={avg_tr:.0%}  PR={avg_pr:.0%}")

    # --- Save results ---
    output_path = Path(__file__).parent / "crag_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "sample_per_domain": sample_size,
                "domains": list(domain_counts.keys()),
                "max_page_chars": max_page_chars,
                "seed": seed,
                "skip_judge": skip_judge,
            },
            "summary": {
                sys_name: {
                    "token_recall": sum(results[sys_name]["token_recall"]) / len(results[sys_name]["token_recall"]),
                    "phrase_recall": sum(results[sys_name]["phrase_recall"]) / len(results[sys_name]["phrase_recall"]),
                    "avg_latency": sum(results[sys_name]["time"]) / len(results[sys_name]["time"]),
                    "llm_judge": (sum(judge_scores.get(sys_name, [])) / len(judge_scores[sys_name])
                                  if judge_scores.get(sys_name) else None),
                }
                for sys_name in ["original", "enhanced", "kb_only"]
            },
            "per_question": per_question,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[*] Results saved to {output_path}")

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CRAG Benchmark Evaluation")
    parser.add_argument("--sample", type=int, default=10,
                        help="Questions per domain (default: 10)")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domains (default: all)")
    parser.add_argument("--max-pages", type=int, default=50_000,
                        help="Max chars per page after HTML strip (default: 50000)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge scoring")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    domains = args.domains.split(",") if args.domains else None

    asyncio.run(evaluate(
        sample_size=args.sample,
        domains=domains,
        max_page_chars=args.max_pages,
        skip_judge=args.no_judge,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
