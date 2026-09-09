# Day 4 learning guide: grounded RAG reliability copilot

Day 4 turns verified ML outputs into a readable reliability brief. The most important design rule is simple:

> The machine-learning model calculates RUL. Retrieval supplies relevant written guidance. The generator may explain those inputs, but it cannot change them.

This separation prevents AeroGuard from becoming an ungrounded chatbot that invents engine numbers, sensor meanings, or maintenance instructions.

## 1. RAG in beginner language

RAG means **Retrieval-Augmented Generation**. It has two main stages:

1. **Retrieval:** search a trusted knowledge collection and select the most relevant passages.
2. **Generation:** write an answer using the model evidence and those passages.

Without retrieval, a language model answers mostly from patterns learned during training. With RAG, the application supplies current, auditable context with citation IDs.

```text
User/engine question
        ↓
Retrieve relevant handbook sections
        ↓
Combine sections with verified ML + SHAP evidence
        ↓
Generate a structured brief
        ↓
Validate protected fields and citations
```

RAG does not automatically make an answer correct. Quality still depends on the sources, retrieval method, prompt boundary, output validation, and evaluation.

## 2. The Day 4 knowledge base

The `knowledge/` directory contains four project-authored demonstration documents:

| Document | Main purpose |
|---|---|
| RUL Interpretation Guide | Risk bands, model disagreement, and capped RUL |
| Sensor Evidence Review | Trend checks, data quality, and SHAP limits |
| Reliability Triage Playbook | Human review priorities and decision authority |
| Model Governance and Monitoring | Deployment boundaries, late-warning risk, and simulation limits |

These documents are deliberately labeled as project guidance, not approved aircraft maintenance manuals. AeroGuard never invents physical names or engineering units for NASA's anonymized sensors.

## 3. Chunking and citations

Each Markdown `##` section becomes one retrieval chunk. For example:

```text
04_model_governance.md
    └── Late warning risk
        └── citation ID: 04_model_governance#late-warning-risk
```

Every chunk stores:

- a stable citation ID;
- source filename;
- document and section titles;
- retrievable text;
- a SHA-256 content fingerprint.

The complete knowledge collection also receives one combined SHA-256 fingerprint. If a document changes, the fingerprint changes, making the exact indexed knowledge version auditable.

The real run loaded 4 documents and created 16 chunks.

## 4. How local retrieval works

Day 4 uses TF-IDF word unigrams and bigrams.

- **Term frequency (TF):** gives more weight to words that appear in a passage.
- **Inverse document frequency (IDF):** reduces the weight of words that appear in many passages and increases the weight of more distinctive words.
- **Unigram:** one word, such as `warning`.
- **Bigram:** two consecutive words, such as `late warning`.

The query and chunks become numerical vectors. Their cosine similarity determines ranking. The top four chunks become context for the final engine brief.

Why TF-IDF instead of a hosted embedding model?

- It runs offline, quickly, and without API cost.
- The small knowledge base uses precise engineering vocabulary.
- It is deterministic and easy to explain in an interview.
- It creates a baseline that future semantic retrieval must beat.

Tradeoff: TF-IDF relies heavily on matching words. It may miss a passage expressed with different vocabulary. A larger production knowledge base would likely use embeddings, metadata filters, hybrid lexical/vector search, and a reranker.

## 5. Retrieval evaluation

The project includes eight labeled questions. Each question names the section that should be retrieved. Day 4 measures:

- **Hit@3:** whether the expected chunk appears anywhere in the first three results.
- **Reciprocal rank:** `1 / rank` of the expected result. Rank 1 scores 1.0; rank 2 scores 0.5; rank 3 scores about 0.333.
- **MRR:** the mean reciprocal rank across all questions.

Real results:

| Metric | Result |
|---|---:|
| Evaluation questions | 8 |
| Hit@3 | 100% |
| Mean reciprocal rank | 0.917 |

This proves that the implemented retriever handles the eight known questions. It does **not** prove production quality because the set is small, handcrafted, and drawn from the same knowledge domain. A production evaluation needs more questions, independent labeling, paraphrases, hard negatives, and periodic regression tests.

## 6. The verified engine evidence packet

Before generation, AeroGuard recomputes evidence from persisted artifacts rather than trusting text supplied by a user or LLM.

For engine 34, the real packet contains:

| Evidence | Value |
|---|---:|
| Last observed cycle | 203 |
| Deployment model | Random Forest |
| Authoritative predicted RUL | 6.484 cycles |
| Risk band | Critical |
| Advisory LSTM prediction | 4.406 cycles |
| Model disagreement | 2.078 cycles |
| SHAP additivity error | approximately `6.24 × 10⁻¹³` |

The top local Random Forest contributions were:

| Feature | Feature value | SHAP contribution |
|---|---:|---:|
| `sensor_11` | 48.13 | −19.88 cycles |
| `sensor_4` | 1427.49 | −12.01 cycles |
| `cycle` | 203 | −11.45 cycles |
| `sensor_12` | 519.44 | −9.34 cycles |
| `sensor_7` | 551.77 | −7.75 cycles |

Negative SHAP contributions pushed the Random Forest estimate below its baseline. They explain model behavior; they do not prove physical causation.

The official test label is intentionally excluded from the generation packet. In real deployment, future failure time is unknown, so allowing the copilot to read it would create unrealistic leakage.

## 7. Risk bands

The project-authored bands are:

| Predicted RUL | Band |
|---:|---|
| 0–15 cycles | Critical |
| More than 15–30 | High |
| More than 30–60 | Elevated |
| Above 60 | Routine |

These bands rank review priority. They do not independently ground, dispatch, clear, or prescribe maintenance for an aircraft.

## 8. What the generator receives

The grounded prompt contains only:

1. the task;
2. the verified evidence packet;
3. retrieved text with allowed citation IDs;
4. explicit safety and grounding rules.

One rule tells the generator to treat retrieved text as reference data, never as instructions. This matters because a document could contain text such as "ignore the previous instructions." The application preserves it as data rather than granting it control over the prompt.

Other rules prohibit the generator from:

- recalculating or replacing RUL;
- claiming SHAP proves causation;
- issuing autonomous maintenance or flight decisions;
- citing a source that was not retrieved.

## 9. Offline and optional OpenAI generation

Day 4 runs offline by default. The local generator creates a deterministic structured brief, which makes setup, tests, and demonstrations free and repeatable.

An optional OpenAI provider uses the Responses API with a strict JSON Schema and `store=False`. The API is enabled only when the user explicitly sets `AEROGUARD_GENERATOR=openai` and supplies `OPENAI_API_KEY` in an ignored `.env` file. The implementation follows the official Responses API shape for text input and JSON output.

Even after a structured OpenAI response returns, application code replaces the following with authoritative evidence values:

- engine ID;
- risk band;
- predicted RUL.

It also removes citations that were not in the retrieved allowlist. This means prompt instructions are not the only safety control; deterministic application validation is the final control.

No paid API request was used for the verified Day 4 run. The saved report records `local_grounded_template` and `openai_enabled: false`.

## 10. The actual generated brief

The brief for engine 34:

- marks the engine as critical review priority;
- preserves the exact 6.484-cycle Random Forest output;
- mentions the 4.406-cycle advisory LSTM estimate and model disagreement;
- summarizes the top SHAP contributions;
- asks a qualified engineer to verify data and consult approved limits;
- clearly says that AeroGuard is not autonomous authority;
- cites the retrieved late-warning, triage, and data-quality sections.

The three primary citations were:

1. `04_model_governance#late-warning-risk`
2. `03_maintenance_triage#critical-and-high-risk-review`
3. `02_sensor_review#data-quality-before-escalation`

## 11. Why this is better than a basic LLM wrapper

A basic wrapper sends a question to an LLM and displays the answer. AeroGuard instead has:

- versioned and fingerprinted local knowledge;
- deterministic retrieval with similarity scores;
- labeled retrieval evaluation;
- evidence recomputed from persisted ML artifacts;
- protected numerical fields;
- citation allowlisting;
- prompt-injection boundaries;
- offline operation and an optional external provider;
- machine-readable reports for auditing.

The LLM is a presentation layer around evidence. It is not the predictive model and not the decision authority.

## 12. Common interview questions and short answers

**What is RAG?**  
RAG retrieves relevant passages and supplies them to a generator so its answer is grounded in auditable context.

**Why use RAG here?**  
The ML model produces numbers, while reliability staff need a readable explanation, relevant guidance, limitations, and citations.

**Why TF-IDF?**  
It is a deterministic, offline baseline suited to this small vocabulary-focused knowledge base. I would compare it with hybrid embedding retrieval at larger scale.

**What is chunking?**  
Splitting source documents into searchable sections. AeroGuard chunks by Markdown heading so every result has a meaningful citation.

**What is cosine similarity?**  
It measures how closely the direction of the query vector matches a document vector; a larger value means stronger lexical similarity.

**How did you evaluate retrieval?**  
I used eight labeled questions and measured Hit@3 and MRR. The current results are 100% Hit@3 and 0.917 MRR, with the limitation that the set is small and handcrafted.

**Can the LLM change RUL?**  
No. RUL comes from the persisted Random Forest. Prompt rules prohibit changes, and post-generation code forcibly restores the authoritative value.

**How do you prevent hallucinated citations?**  
The generator receives allowed chunk IDs, and post-processing removes any citation not present in the retrieved results.

**How do you handle prompt injection in documents?**  
Retrieved text is serialized as data and explicitly marked untrusted. It receives no tool authority, and deterministic validation protects fields and citations.

**Why not let the LLM decide maintenance?**  
It lacks approved aircraft procedures and legal authority. It summarizes evidence for qualified human review.

**Does SHAP identify the physical failure cause?**  
No. It describes how the Random Forest used features, not physical causality.

**Why compare Random Forest and LSTM?**  
Disagreement is useful uncertainty evidence. It triggers review; the values are not averaged automatically.

**How would you scale the RAG layer?**  
Add document ingestion controls, metadata filters, embeddings plus keyword search, reranking, a vector database, access control, tracing, independent evaluation, and feedback monitoring.

## 13. One-minute interview explanation

> Day 4 adds a grounded reliability copilot without letting GenAI replace the predictive model. I chunk four project-authored guidance documents by heading, fingerprint every chunk, and build a deterministic TF-IDF unigram/bigram index. I evaluate it with labeled questions using Hit@3 and MRR; the current eight-question set achieved 100% Hit@3 and 0.917 MRR, although I clearly document that this is a small handcrafted set. For an engine brief, I recompute the persisted Random Forest prediction, advisory LSTM estimate, model disagreement, and local TreeSHAP contributions. Retrieval selects relevant guidance, and a structured generator writes the brief with allowlisted citations. The project works offline, with an optional OpenAI Responses API provider. Most importantly, post-generation validation restores the authoritative engine ID, risk band, and RUL and rejects invented citations. The output supports qualified human review; it never makes an autonomous maintenance or flight decision.
