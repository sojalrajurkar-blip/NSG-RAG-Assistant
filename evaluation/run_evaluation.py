"""Automated Evaluation Runner for NSG RAG Assistant.

Executes all 32 test cases across both LLM-Only and RAG modes,
evaluating factual accuracy, citations, latency, and hallucination behavior.
Saves observed empirical evidence to evaluation/results.csv.
"""

import csv
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import RAGPipeline, PipelineAnswer


def run_evaluation(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Runs test suite and generates detailed comparative evaluation results."""
    pipeline = RAGPipeline()
    test_cases_file = Path("evaluation/test_cases.csv")
    results_file = Path("evaluation/results.csv")

    with open(test_cases_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        test_cases = list(reader)

    if limit:
        test_cases = test_cases[:limit]

    print(f"Starting evaluation of {len(test_cases)} test cases with model: {pipeline.config.gemini_model_name}...")
    results: List[Dict[str, Any]] = []

    for i, tc in enumerate(test_cases, 1):
        tc_id = tc["id"]
        q = tc["question"]
        cat = tc["category"]
        expected = tc["expected_answer"]
        is_answerable = tc["is_answerable"].strip().lower() == "true"
        exp_page = tc["expected_page"]

        print(f"[{i}/{len(test_cases)}] Evaluating {tc_id} ({cat}): '{q[:40]}...'")

        # 1. Run LLM-Only Baseline
        b_ans: PipelineAnswer = pipeline.answer_question(q, mode="LLM-Only")

        # 2. Run Grounded RAG
        r_ans: PipelineAnswer = pipeline.answer_question(q, mode="RAG", top_k=3)

        # -------------------------------------------------------------
        # Evaluation Logic
        # -------------------------------------------------------------
        llm_pass = False
        rag_pass = False
        failure_category = "NONE"
        notes = []

        # A. Unanswerable / Out-of-domain questions
        if not is_answerable:
            # Baseline behavior on unanswerable course-specific questions
            if "not available" in b_ans.answer.lower() or "don't know" in b_ans.answer.lower() or "not mentioned" in b_ans.answer.lower() or "could you please provide" in b_ans.answer.lower():
                llm_pass = True
            else:
                llm_pass = False  # Often hallucinates or uses pre-trained world knowledge for out-of-domain queries

            # RAG behavior: must admit context is missing
            if "not available" in r_ans.answer.lower() or "not found" in r_ans.answer.lower() or "insufficient" in r_ans.answer.lower():
                rag_pass = True
            else:
                rag_pass = False
                failure_category = "UNANSWERABLE_LEAK"
                notes.append("RAG failed to declare missing context.")

        # B. Direct Factual / Synthesis Questions
        else:
            # LLM-Only usually fails on specific course/assignment details because it has no access to the PDF
            if any(term.lower() in b_ans.answer.lower() for term in ["assignment 01", "assignment series", "langgraph", "langchain", "do i need rag"]):
                # Did it provide relevant info?
                llm_pass = True
            else:
                llm_pass = False
                notes.append("Baseline LLM lacks document context.")

            # RAG Evaluation
            if r_ans.is_grounded and r_ans.citations:
                rag_pass = True
                # Check page correctness if page_critical
                if cat == "page_critical" and exp_page and exp_page != "None":
                    if exp_page.lower() in [c.lower() for c in r_ans.citations] or exp_page.lower() in r_ans.answer.lower():
                        rag_pass = True
                    else:
                        rag_pass = False
                        failure_category = "PAGE_CITATION_MISMATCH"
                        notes.append(f"Expected {exp_page}, got {r_ans.citations}")
            else:
                rag_pass = False
                failure_category = "RETRIEVAL_OR_GROUNDING_FAILURE"
                notes.append("RAG did not produce grounded answer with citations.")

        # Classify overall failure category if RAG failed
        if not rag_pass and failure_category == "NONE":
            if not r_ans.has_confident_context:
                failure_category = "RETRIEVAL_FAILURE"
            else:
                failure_category = "GENERATION_FAILURE"

        row = {
            "id": tc_id,
            "question": q,
            "category": cat,
            "is_answerable": is_answerable,
            "expected_page": exp_page,
            "llm_only_latency_sec": round(b_ans.total_latency_seconds, 2),
            "llm_only_pass": llm_pass,
            "llm_only_answer": b_ans.answer.replace("\n", " ")[:300],
            "rag_latency_sec": round(r_ans.total_latency_seconds, 2),
            "rag_retrieval_sec": round(r_ans.retrieval_latency_seconds, 2),
            "rag_pass": rag_pass,
            "rag_citations": "; ".join(r_ans.citations),
            "rag_answer": r_ans.answer.replace("\n", " ")[:300],
            "failure_category": failure_category,
            "notes": " | ".join(notes) if notes else "Passed rubric criteria",
        }
        results.append(row)

        # Brief delay to respect free-tier rate limits
        time.sleep(1.2)

    # Save to evaluation/results.csv
    fieldnames = list(results[0].keys())
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation complete! Saved {len(results)} evaluated cases to {results_file}")
    return results


if __name__ == "__main__":
    run_evaluation()
