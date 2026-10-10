# Empirical Failure Log: LLM-Only Baseline vs. Grounded RAG

This log documents **7 real observed failures** captured during the 32-case automated evaluation of the NSG Academy AI Engineering PDF assistant (`NSG_Product_Engineering_AI_Assignment_Series_2026_STUDENT.pdf`).

---

## Summary of Observed Failure Categories

| Category | Frequency | Primary Mode Affected | Root Cause |
| :--- | :---: | :---: | :--- |
| **Complete Context Absence** | 22 | LLM-Only | Model has no access to private course PDFs; cannot answer program-specific queries. |
| **Hallucination / Guessing** | 4 | LLM-Only | Generates believable but fictitious assignments when prompted with generic titles. |
| **Transient Rate Limit (HTTP 429)** | 2 | RAG | Google AI Studio free tier enforces requests-per-minute limits during automated batch runs. |
| **Semantic Drift / Retrieval Miss** | 1 | RAG | Short, generic keyword queries ("Page 1 heading") retrieved topically dense pages (Pages 5, 12, 4) rather than title page. |
| **Edge Case Input Handling** | 1 | RAG & Baseline | Whitespace-only input triggers immediate input validation guardrail. |

---

## Detailed Failure Case Studies

### 1. Case F-01 (TC-01): Complete Context Absence in LLM-Only
- **Question:** *"What is the primary goal of Assignment 01?"*
- **Mode:** LLM-Only Baseline
- **Observed Behavior:** The model answered:  
  > *"It looks like you didn't mention which course, class, or program your 'Assignment 01' is for! Could you please provide a little more context..."*
- **Expected Behavior:** Technology selection & problem framing; learning when RAG is needed before touching a vector DB.
- **Root Cause:** Fundamental limitation of foundation models. Pre-trained models cannot answer private organizational questions without retrieval or fine-tuning.
- **RAG Comparison:** In RAG mode, the assistant retrieved Page 2 and Page 13, answering with 100% precision and citing `[Page 2]`.

---

### 2. Case F-02 (TC-07): Unsupported Course Synthesis in Baseline
- **Question:** *"What are the five assignments in this series in order?"*
- **Mode:** LLM-Only Baseline
- **Observed Behavior:** The baseline generated generic software engineering assignments (e.g., "1. Introduction to AI, 2. Neural Networks, 3. Computer Vision, 4. NLP, 5. Capstone Project").
- **Expected Behavior:** Assignment 01 (Do I Need RAG?), Assignment 02 (LangChain), Assignment 03 (Chain/Workflow/Agent), Assignment 04 (Prove It), Assignment 05 (Production Product).
- **Root Cause:** Hallucination under uncertainty. The model attempted to satisfy the prompt by extrapolating from generic training data.
- **RAG Comparison:** RAG retrieved the syllabus summary from Page 2 and correctly enumerated all 5 assignments verbatim with exact titles and page citations.

---

### 3. Case F-03 (TC-17): Transient Rate Limit Exhaustion (HTTP 429)
- **Question:** *"What technology matrix must be completed in Assignment 05?"*
- **Mode:** Grounded RAG
- **Observed Behavior:** Output error: `Rate limit exceeded (HTTP 429). Please wait a moment before trying again.`
- **Expected Behavior:** Detailed technology vs approach justification matrix from Page 12.
- **Root Cause:** The free-tier Gemini API key has a quota limit of 15 requests per minute (RPM). Running 32 sequential tests with dual calls (64 total API requests) exceeded the per-minute threshold during mid-run execution.
- **Mitigation / Fix:** Implemented exponential backoff retry in `LLMService` and inter-request sleep pacing in the evaluation runner.

---

### 4. Case F-04 (TC-27): Semantic Drift & Keyword Retrieval Miss
- **Question:** *"What specific section heading appears on Page 1?"*
- **Mode:** Grounded RAG
- **Observed Behavior:** The retriever returned chunks from Pages 5, 12, and 4 instead of Page 1. Consequently, the grounded LLM honestly stated:  
  > *"Based on the provided document excerpts, this information is not available."*
- **Expected Behavior:** *"NSG ACADEMY PRODUCT ENGINEERING AI ENGINEERING DECISION & PRODUCT BUILDING"* from Page 1.
- **Root Cause:** Vector embedding models (`all-MiniLM-L6-v2`) represent semantic meaning, not page numbers. When asked for "section heading", pages with dense headings (like Page 5 and 12) had higher cosine similarity than the cover page.
- **Insight:** Pure semantic search struggles with structural metadata queries (e.g., "on Page 1"). A hybrid search (metadata filter + semantic vector) is necessary for structural queries.

---

### 5. Case F-05 (TC-29): Whitespace Edge Case
- **Question:** `"   "` (Whitespace string)
- **Mode:** RAG & Baseline
- **Observed Behavior:** Both modes returned immediate validation message: `No question provided.`
- **Expected Behavior:** Clean validation without unhandled crash or exception.
- **Root Cause / Assessment:** Handled cleanly as an early return. The test rubric marked this as a strict format difference, demonstrating the necessity of robust input sanitation before touching embedding models.

---

### 6. Case F-06 (TC-21): Out-of-Domain World Knowledge Leak
- **Question:** *"Who is the current President of the United States in 2026?"*
- **Mode:** LLM-Only vs RAG Comparison
- **Observed Behavior in Baseline:** Baseline answered using pre-trained historical political knowledge.
- **Observed Behavior in RAG:** RAG system prompt successfully enforced the guardrail:  
  > *"Based on the provided document excerpts, this information is not available."*
- **Root Cause & Significance:** Proves that the conservative academic system prompt successfully prevents the model from leaking external out-of-domain knowledge when tasked with PDF-grounded Q&A.

---

### 7. Case F-07 (TC-10): Checklist Artifact Absence in Baseline
- **Question:** *"What artifacts are required in the Master Submission Checklist for every assignment?"*
- **Mode:** LLM-Only Baseline
- **Observed Behavior:** Baseline gave a generic software checklist (e.g., "README, unit tests, Dockerfile, Git repo").
- **Expected Behavior:** Program-specific checklist: working code, test evidence, failure log, evaluation report, and architectural justification (from Page 14).
- **Root Cause:** Private operational specifications cannot be guessed by public LLMs.
- **RAG Comparison:** RAG retrieved Page 14 verbatim, citing `[Page 14]` and listing the exact 5 mandatory submission artifacts.
