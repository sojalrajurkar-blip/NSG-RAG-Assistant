# NSG Academy Knowledge Assistant — Complete Antigravity Build Instructions

## 1. Project goal

Build a local-first PDF Question Answering RAG application for the NSG Academy AI Engineering Assignment Series.

The application must answer questions using information retrieved from uploaded PDF documents, show source document/page information when available, and provide a way to compare:

1. **LLM-only baseline** — send the question directly to the language model without retrieved document context.
2. **RAG** — retrieve relevant PDF chunks and provide those chunks as context to the language model.

Do not assume RAG is always better. Collect actual evidence, evaluate both approaches, document failures, and make a reasoned decision.

This project is intended to support learning and Assignment 1. It should also establish useful testing habits for the later assignments, but do not claim that this small project automatically completes all five assignments.

## 2. Mandatory collaboration and approval rules

**These rules are mandatory for every phase and every significant subtask.**

### Before starting any phase

Do not immediately edit or create code. First send me a short **Phase Start Brief** containing:

1. **What we are doing:** explain the current phase in simple language.
2. **Why it is needed:** connect it to the RAG pipeline and assignment.
3. **What will be created or changed:** list exact files, folders, configuration, or commands.
4. **What is required:** software, API access, keys, documents, permissions, or user decisions.
5. **How to get it:** provide beginner-friendly, step-by-step instructions for anything I need to install, download, configure, or obtain.
6. **What I need to do:** clearly separate my tasks from your tasks.
7. **Risks and costs:** mention API usage, downloads, secrets, destructive changes, or other relevant risks.
8. **How we will verify it:** state the commands/tests and expected evidence of success.
9. **Approval question:** ask, “Shall I proceed with this phase?” and wait.

Do not begin that phase until I explicitly approve it.

### During a phase

- Work only on the approved phase and its agreed scope.
- If a new dependency, API key, account, document, permission, or design decision becomes necessary, stop and explain what is needed, why it is needed, and how I can obtain it. Ask me before proceeding if my action or decision is required.
- Do not silently choose a paid service, create an account, change the technology stack, or introduce extra frameworks.
- Never display, print, commit, or log secret API keys.
- Do not run destructive commands or overwrite existing files without explaining the impact and receiving approval.
- Keep the implementation beginner-friendly and modular.
- Do not fabricate test results, sample measurements, failures, citations, or successful commands.
- If a command fails, explain the actual error in simple language, diagnose it, propose a fix, and ask before making a significant change.
- If you cannot verify something, say so clearly.

### At the end of every phase

Provide a **Phase Completion Report** containing:

1. What was completed.
2. Exact files created/modified.
3. Commands actually run.
4. Test results and evidence, distinguishing passed, failed, skipped, and not run.
5. Any remaining issue or limitation.
6. What I should understand from this phase, in simple language.
7. What the next phase will do and why.
8. Requirements for the next phase and step-by-step instructions to obtain them.
9. Ask, “Shall we proceed to Phase N?” and wait for my explicit approval.

**Never automatically start the next phase.** My approval applies only to the phase I approved; it is not blanket approval for the whole project.

## 3. Technology choices

Use this initial stack unless a real compatibility issue appears. If a change is needed, explain the reason and ask me before switching.

- **Python** — application logic.
- **Streamlit** — simple local web interface.
- **PyMuPDF** — extract text from PDFs and retain page numbers.
- **Sentence Transformers** with `all-MiniLM-L6-v2` initially — create text embeddings locally.
- **ChromaDB** — store embeddings, text chunks, and metadata; support similarity search.
- **Google Gen AI Python SDK (`google-genai`)** — call a Gemini model for answer generation.
- **python-dotenv** — load local environment variables.
- **pytest** — automated tests.
- Python standard library where practical.

Do not use LangChain in the first implementation. Assignment 2 requires a separate comparison between direct API usage and LangChain, so keep the first version based on direct API calls.

Do not train a foundation model from scratch. This project is a RAG application using an existing embedding model and an existing LLM.

Before choosing a Gemini model, verify which model is available to the configured API key/account. Keep the model name configurable in `.env`; do not hardcode an assumed model name throughout the code.

## 4. Expected project structure

Propose this structure during Phase 1 and explain it before creating files. You may recommend a small, justified adjustment, but ask first.

```text
NSG-RAG-Assistant/
├── app.py
├── .env.example
├── .env                 # local only; never commit
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── documents/       # local source PDFs; do not commit private documents
├── chroma_db/           # generated persistent vector index; normally ignored by Git
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm_service.py
│   └── rag_pipeline.py
├── tests/
│   ├── test_pdf_loader.py
│   ├── test_chunker.py
│   ├── test_retriever.py
│   ├── test_rag_pipeline.py
│   └── test_rag.py
└── evaluation/
    ├── test_cases.csv
    ├── results.csv
    ├── failure_log.md
    ├── incident_report.md
    └── final_evaluation_report.md
```

Avoid duplicate or unnecessary files. Explain any structural changes before applying them.

## 5. Development phases — complete roadmap

There are **10 development phases**. Each phase must follow the approval rules above. Each phase may have multiple steps; do not skip verification.

### Phase 1 — Workspace inspection, setup, and environment

**Goal:** Prepare a clean, working Python environment without damaging existing files.

Steps:
1. Inspect the current workspace and report existing files before changing anything.
2. Check Python and pip versions.
3. Confirm the working directory and whether files already exist.
4. Propose the folder structure.
5. Explain virtual environments and why this project uses one.
6. After approval, create/activate `.venv` using commands suitable for my operating system.
7. Install the agreed dependencies only after explaining them.
8. Create `requirements.txt`, `.gitignore`, and `.env.example`.
9. Explain how to create `.env` locally and why it must not be committed.
10. Verify imports and dependency versions.

Windows command examples, to be used only after approval and after checking the workspace:

```bat
cd D:\ProjectFolder
mkdir NSG-RAG-Assistant
cd NSG-RAG-Assistant
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Do not assume the folder does not already exist. If it exists, inspect it first.

Acceptance checks:
- Python environment activates successfully.
- Dependencies install without unresolved errors.
- `.env` and generated local data are ignored by Git.
- No API key has been added to source control.

Stop and provide the Phase 1 Completion Report. Ask for approval before Phase 2.

### Phase 2 — PDF ingestion and text extraction

**Goal:** Read a PDF and extract useful text while retaining its source and page number.

Steps:
1. Explain which PDF will be used and where it should be placed.
2. Ask me to provide or approve the source PDF if it is not already available.
3. Implement `src/pdf_loader.py` using PyMuPDF.
4. Extract text page by page.
5. Store document name, page number, and extracted text.
6. Handle invalid PDFs, empty pages, unreadable files, and extraction errors clearly.
7. Detect likely scanned/image-only pages and explain that OCR may be needed; do not add OCR dependencies without approval.
8. Add tests using a small test PDF or an appropriately created fixture.
9. Show a safe preview of extracted text and page metadata without dumping an unnecessarily large document.

Acceptance checks:
- Extracted text can be traced back to a document and page.
- Empty or invalid PDFs produce clear errors.
- Tests and manual verification results are honestly reported.

Stop and ask for approval before Phase 3.

### Phase 3 — Chunking and metadata

**Goal:** Split extracted text into useful, overlapping chunks without losing provenance.

Steps:
1. Explain why a whole PDF is usually too large and imprecise to retrieve as one unit.
2. Implement `src/chunker.py`.
3. Start with an experimental target around 500 tokens per chunk and around 80 tokens overlap. Explain that these are initial values to evaluate, not universal best settings.
4. Preserve document name, page number(s), chunk ID, and any useful section heading that can be reliably extracted.
5. Handle short pages, blank text, and chunks crossing page boundaries transparently.
6. Add tests for empty input, short text, long text, overlap, stable IDs, and metadata.
7. Display a few example chunks for manual inspection.

Acceptance checks:
- Chunks are neither empty nor unintentionally duplicated.
- Source metadata remains attached.
- Chunk size and overlap can be configured rather than buried in code.

Stop and ask for approval before Phase 4.

### Phase 4 — Embeddings and persistent ChromaDB index

**Goal:** Convert chunks into vectors and store them so relevant information can be searched.

Steps:
1. Explain embeddings and vector databases in simple language.
2. Implement `src/embeddings.py` with Sentence Transformers using `all-MiniLM-L6-v2` initially.
3. Explain that the first model download requires an internet connection and disk space; ask before downloading if required by the environment.
4. Implement `src/vector_store.py` using ChromaDB.
5. Store chunk text, embedding, stable ID, and metadata.
6. Persist the index locally and document how to rebuild it when documents or settings change.
7. Prevent accidental duplicate indexing when the same document is ingested again.
8. Add a way to inspect index count and sample metadata.
9. Test vector dimensions, persistence/reload, duplicate handling, and empty input.

Acceptance checks:
- Embeddings are generated successfully.
- The index can be reloaded after the program restarts.
- Chunk text and source metadata are retrievable.
- The app does not claim that a database stores or generates answers; answer generation is the LLM's job.

Stop and ask for approval before Phase 5.

### Phase 5 — Semantic retrieval

**Goal:** Retrieve the most relevant document chunks for a user question.

Steps:
1. Implement `src/retriever.py`.
2. Embed the question using the same compatible embedding model.
3. Search ChromaDB for the nearest chunks.
4. Start with configurable `top_k = 3`; treat it as an experimental value.
5. Return chunk text, source document, page number, IDs, and available similarity/distance metadata.
6. Explain that vector similarity is a relevance signal, not proof that a chunk answers the question.
7. Provide a debug view showing retrieved chunks before calling the LLM.
8. Handle empty questions, an empty index, low-relevance retrieval, and missing metadata.
9. Build retrieval tests using known questions and expected source pages where possible.

Acceptance checks:
- Relevant questions retrieve useful source chunks in the test set.
- Retrieved chunks and source details can be inspected.
- Unanswerable questions and low-confidence retrieval are not silently treated as successful retrieval.

Stop and ask for approval before Phase 6.

### Phase 6 — Gemini API integration

**Goal:** Add answer generation through a configured Gemini model.

Steps:
1. Explain what an API key is, how to obtain it from Google AI Studio, and that access, quotas, billing, and model availability may vary.
2. Do not ask me to paste a secret into chat. Guide me to store it locally in `.env`.
3. Verify available model names using the current supported SDK/API method where possible. Do not invent availability.
4. Implement `src/config.py` and `src/llm_service.py`.
5. Read secrets from environment variables; never print them.
6. Keep model name, timeouts, and other appropriate settings configurable.
7. Implement a direct LLM-only function that sends just the question.
8. Implement a context-grounded generation function that sends the question and retrieved context.
9. Use a conservative instruction: answer from supplied context, admit when context is insufficient, and include source/page metadata when provided.
10. Handle rate limits, network errors, invalid keys, unavailable models, and empty responses.
11. Add mocked unit tests so most tests do not require paid API calls. Clearly label live API tests as such.

Acceptance checks:
- API key is not in source files, logs, screenshots, or Git.
- A controlled live test works only if the user has configured access.
- API failures are handled and reported honestly.

Stop and ask for approval before Phase 7.

### Phase 7 — Complete RAG pipeline

**Goal:** Connect ingestion, chunking, embeddings, retrieval, and generation.

Steps:
1. Implement `src/rag_pipeline.py`.
2. Define clear functions for indexing documents, answering a RAG question, and answering an LLM-only question.
3. Make the RAG flow: question → retrieval → context construction → LLM generation → answer and sources.
4. Return a structured result containing answer, source references, retrieved chunks, and relevant timing/diagnostic data.
5. Keep indexing separate from question answering so PDFs are not re-embedded on every question.
6. Provide a clear answer when no suitable context is found; do not fabricate a source.
7. Ensure retrieved PDF text is treated as untrusted reference content, not instructions that can override system rules.
8. Add unit/integration tests with mocked LLM calls and a small controlled document fixture.
9. Document the architecture and data flow.

Acceptance checks:
- End-to-end RAG works on a controlled sample.
- The answer and its cited source can be traced to retrieved chunks.
- Missing context and API errors are handled explicitly.

Stop and ask for approval before Phase 8.

### Phase 8 — Streamlit user interface

**Goal:** Provide a simple, usable local interface.

Steps:
1. Explain what the UI will contain before implementing it.
2. Implement `app.py` with:
   - document upload or selection of approved local PDFs;
   - an explicit indexing action and indexing status;
   - a question input and Ask button;
   - answer display;
   - source document and page details;
   - an expandable retrieved-chunks/debug section;
   - clear loading, empty, success, and error states;
   - a way to choose LLM-only or RAG for comparison.
3. Keep API keys out of the UI.
4. Validate file type and handle upload/processing errors.
5. Avoid unnecessary UI complexity and avoid adding chatbot gimmicks.
6. Add basic UI/manual test instructions.
7. Verify how local uploads and ChromaDB persistence behave on app reruns.

Acceptance checks:
- The app starts with the documented command.
- Users can index a document, ask a question, and inspect sources.
- Both LLM-only and RAG modes can be selected.
- Errors are understandable to a beginner.

Stop and ask for approval before Phase 9.

### Phase 9 — Evaluation: compare LLM-only and RAG

**Goal:** Collect evidence showing where RAG helps, does not help, or fails.

Steps:
1. Explain the evaluation plan before creating the dataset.
2. Prepare at least 30 test cases for Assignment 4 readiness, including:
   - direct factual questions;
   - questions needing multiple chunks;
   - paraphrased questions;
   - ambiguous questions;
   - unanswerable questions;
   - missing-document and retrieval edge cases;
   - questions where source/page correctness matters.
3. Each case should include ID, question, expected answer or rubric, answerability, expected source/page where known, LLM-only result, RAG result, pass/fail, notes, and failure category.
4. Run both modes on the same questions with the same model/settings where practical. Record date, model, prompt/version, retrieval settings, and any relevant limitations.
5. Define evaluation criteria before looking at the results. Evaluate factual correctness, context/source support, answerability behavior, retrieval relevance, and latency. Do not rely only on a vague overall score.
6. Distinguish retrieval failure from generation failure.
7. Save actual results in `evaluation/results.csv`.
8. Analyze at least 7 real failures if they occur. Never invent failures merely to reach a number. If fewer than 7 failures occur, create additional stress/edge-case tests and clearly distinguish observed failures from intentionally constructed test scenarios.
9. For one important real failure, prepare an incident report: impact, reproduction steps, expected vs actual behavior, root cause, fix, regression test, and remaining risk.
10. Make an improvement, rerun relevant tests, and compare before/after results.
11. Discuss trade-offs: retrieval quality, latency, complexity, embedding/index maintenance, and API cost where measurable.
12. State limitations honestly; do not claim RAG is better unless the evidence supports it.

Acceptance checks:
- At least 30 documented test cases exist.
- Results are actual observed outputs or clearly marked simulated/mock results.
- Failure analysis and incident report are traceable to evidence.
- The final decision is reasoned, not predetermined.

Stop and ask for approval before Phase 10.

### Phase 10 — Documentation and final handover

**Goal:** Make the project reproducible and prepare the assignment evidence.

Steps:
1. Complete `README.md` with prerequisites, Windows setup, activation, installation, environment variables, indexing, running the app, tests, troubleshooting, and cleanup.
2. Verify `requirements.txt` matches the tested environment.
3. Document architecture and the end-to-end data flow.
4. Record design decisions and alternatives, including why direct API was used initially and why RAG was or was not justified for this use case.
5. Complete the evaluation report, failure log, and incident report based on actual evidence.
6. Explain how to reproduce tests and where results are stored.
7. Provide a list of known limitations and sensible future improvements.
8. Prepare a 10–15 minute presentation outline for the overall AI product learning journey, while clearly identifying which Assignment 5 requirements remain unimplemented.
9. Run the final test/build checks and report exactly what passed, failed, or was not run.
10. Provide a final file tree and handover summary.

Acceptance checks:
- A new user can follow the README to run the application.
- No secrets or private documents are accidentally included in version control.
- Reports reflect actual implementation and test evidence.
- Assignment 1 conclusions are supported by the collected results.

After Phase 10, provide the final completion report. Do not claim that Assignment 5 is complete unless its original PDF requirements have been separately checked and satisfied.

## 6. Required security and reliability rules

- Never commit `.env`, API keys, tokens, private PDFs, or sensitive user data.
- Include `.env.example` with placeholder names only, never real secrets.
- Do not log API keys or full sensitive document contents.
- Validate file extensions and handle malformed files.
- Set sensible limits for file size and input length; explain the chosen values.
- Handle API rate limits and network errors.
- Treat retrieved content as untrusted input.
- Do not claim answers are guaranteed correct.
- Keep dependencies minimal.
- Ask before adding OCR, reranking, hosted vector databases, paid services, authentication, deployment, or any new major dependency.

## 7. Required explanation style

I am learning and need beginner-friendly guidance. Use simple English in code/comments where appropriate, and explain each phase to me in simple Marathi or simple English before starting.

For every command:
- tell me where to run it (Command Prompt/PowerShell/terminal);
- explain what it does;
- show the expected result;
- explain common errors and what to do.

Do not overwhelm me with all implementation details at once. The complete roadmap is provided here so you can plan ahead, but reveal and execute only the current approved phase.

## 8. First response required from Antigravity

Do **not** create or edit project files yet.

Your first response must:
1. Confirm that you understand the project and the mandatory approval workflow.
2. Inspect the existing workspace without modifying files, if workspace access is available.
3. Tell me what Phase 1 will accomplish.
4. List the exact requirements for Phase 1 and how to obtain/check them.
5. Explain the proposed folder structure.
6. Show the verification commands you intend to use.
7. Identify anything you cannot inspect or verify.
8. Ask: **“Shall I proceed with Phase 1?”**

Then wait for my answer. Do not start implementation until I approve Phase 1.
