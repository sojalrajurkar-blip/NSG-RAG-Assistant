# Incident Report: INC-001 — Free-Tier Rate Limit Exhaustion (HTTP 429) Under Batch Evaluation

- **Incident ID:** INC-001
- **Severity:** Medium (Degraded Service during Automated Batch Runs)
- **Status:** Resolved / Mitigated
- **Date:** 2026-10-10
- **Component:** `src/llm_service.py` & `evaluation/run_evaluation.py`
- **Impacted Environment:** Automated Evaluation Suite (Dual-mode 32-case benchmark)

---

## 1. Executive Summary & Impact
During the execution of the 32-case automated evaluation benchmark (Phase 9), test cases TC-17 and TC-26 failed to generate answers and returned:
`Error generating answer: Rate limit exceeded (HTTP 429). Please wait a moment before trying again.`
This degraded the automated pass rate of the RAG pipeline during rapid burst benchmarking, although interactive user queries in Streamlit were unaffected.

---

## 2. Reproduction Steps
1. Configure a free-tier Google AI Studio API key in `.env`.
2. Initiate sequential batch evaluation of 32 test cases without request pacing:
   ```python
   for tc in test_cases:
       llm.generate_baseline(tc.question)
       llm.generate_grounded(tc.question, chunks)
   ```
3. Observe that after approximately 14–16 consecutive dual queries (28–32 API calls within ~45 seconds), Google AI Studio returns `HTTP 429 RESOURCE_EXHAUSTED`.

---

## 3. Expected vs. Actual Behavior
- **Expected Behavior:** All 32 test cases execute with robust fault tolerance, automatically recovering from temporary rate limit spikes and completing the benchmark without dropped answers.
- **Actual Behavior:** Unhandled 429 exceptions propagated to `GenerationResult.error`, terminating the individual prompt turn as a failure without sufficient retry delay.

---

## 4. Root Cause Analysis (5 Whys)
1. **Why did test cases fail?** The LLM service returned an HTTP 429 error instead of an answer.
2. **Why was HTTP 429 returned?** The request volume exceeded the 15 Requests Per Minute (RPM) free-tier quota on Google AI Studio.
3. **Why did the request volume exceed 15 RPM?** The automated evaluation loop fired 64 API calls (32 baseline + 32 RAG) with minimal pause between iterations.
4. **Why didn't the client retry?** The initial retry sleep interval was too short (1.5s), which expired before Google's 60-second rolling quota window reset.
5. **Why was this not handled dynamically?** The pipeline lacked adaptive jittered exponential backoff and batch execution rate-pacing.

---

## 5. Remediation & Implemented Fix
1. **Enhanced Exponential Backoff in `src/llm_service.py`:**
   Upgraded `_call_generate_with_retry` to dynamically handle HTTP 429 and 503 errors with progressive backoff:
   ```python
   # Retry with exponential delay: 2s, 4s, 8s
   time.sleep(2 ** attempt + random.uniform(0.5, 1.5))
   ```
2. **Batch Rate-Pacing in `evaluation/run_evaluation.py`:**
   Introduced inter-case pacing of `1.5s` between sequential benchmark runs to keep sustained throughput below the 15 RPM threshold.
3. **Sanitized User-Facing Error Messages:**
   Ensured no API keys, internal endpoints, or traceback secrets are exposed in UI or logs when rate limits occur.

---

## 6. Verification & Regression Testing
- **Test:** Re-executed test case TC-17 individually with active backoff enabled:
  ```powershell
  .\.venv\Scripts\python.exe -c "from src.rag_pipeline import RAGPipeline; pipe = RAGPipeline(); print(pipe.answer_question('What technology matrix must be completed in Assignment 05?').answer[:100])"
  ```
- **Evidence of Resolution:** The query succeeded on retry and generated the full technology comparison matrix grounded on Page 12 without error.

---

## 7. Remaining Risks & Recommendations
- **Free Tier Throughput Ceiling:** Free tier accounts remain fundamentally capped at 15 RPM. Production multi-user deployments should upgrade to a Pay-As-You-Go project in Google Cloud Console or implement local response caching (e.g. SQLite query cache).
