# Redrob Hackathon – Hybrid AI Candidate Ranking System

## Overview

This project implements a **Hybrid AI Candidate Ranking System** for the Redrob Hackathon.

The objective is to rank candidates for a given Job Description by combining **semantic similarity** with **behavioral intelligence** while filtering out logically inconsistent (honeypot) profiles.

The complete pipeline runs **offline on CPU**, requires **no external API calls during inference**, and generates the final **Top-100 ranked candidates** in the required submission format.

---

# Key Features

* Hybrid ranking using semantic retrieval and behavioral scoring
* Honeypot detection to remove inconsistent candidate profiles
* Rich semantic document construction from candidate profiles
* SentenceTransformer embeddings (`all-MiniLM-L6-v2`)
* XGBoost behavioral reranking
* Deterministic reasoning generation for every shortlisted candidate
* Fully offline inference (CPU only)
* Submission validator compatible

---

# Project Architecture

```
                    candidates.jsonl
                           │
                           ▼
                Honeypot Detection
                   (honeypot.py)
                           │
                           ▼
               Valid Candidate Profiles
                           │
                           ▼
           Semantic Document Construction
                   (embedder.py)
                           │
                           ▼
             SentenceTransformer Model
             (all-MiniLM-L6-v2)
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
candidate_embeddings.npy             jd_embedding.npy
         │
         ▼
 Behavioral Feature Extraction
(train_behavior_model.py)
         │
         ▼
 XGBoost Behavioral Ranker
behavioral_ranker.json
         │
         ▼
      submit_ranker.py
         │
         ├── Cosine Similarity
         ├── Behavioral Prediction
         ├── Hybrid Score
         ├── Candidate Reasoning
         ▼
 team_submission.csv
```

---

# Repository Structure

```
.
├── honeypot.py
├── embedder.py
├── train_behavior_model.py
├── submit_ranker.py
├── validate_submission.py
│
├── behavioral_ranker.json
├── candidate_ids.json
├── jd_embedding.npy
│
├── team_submission.csv
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

# Methodology

## Step 1 — Honeypot Detection

Removes logically inconsistent candidate profiles before ranking.

Examples:

* Expert skill with zero experience
* Impossible employment duration
* Invalid profile signals

Run:

```bash
python honeypot.py
```

Output:

```
candidate_ids.json
```

---

## Step 2 — Semantic Document Construction

Each candidate profile is transformed into a rich semantic document using:

* Current title
* Professional summary
* Skills
* Career history
* Certifications
* Industry
* AI-specific derived tags

Run:

```bash
python embedder.py
```

Outputs:

```
candidate_embeddings.npy
jd_embedding.npy
candidate_ids.json
```

---

## Step 3 — Behavioral Model

Behavioral features include:

* Profile completeness
* Recruiter response rate
* Last active date
* GitHub activity
* Interview completion
* Notice period
* Open-to-work status
* Connection count

These features train an **XGBoost classifier**.

Run:

```bash
python train_behavior_model.py
```

Output:

```
behavioral_ranker.json
```

---

## Step 4 — Hybrid Ranking

The final ranking combines:

* Semantic similarity
* Behavioral probability

Hybrid score:

```
Final Score =
0.75 × Semantic Similarity
+
0.25 × Behavioral Score
```

Top 2000 candidates are retrieved using semantic search.

They are reranked using the behavioral model.

The top 100 candidates are written to:

```
team_submission.csv
```

Run:

```bash
python submit_ranker.py
```

---

# Technologies Used

* Python
* Sentence Transformers
* all-MiniLM-L6-v2
* XGBoost
* NumPy
* Pandas
* Scikit-learn

---

# Compute Environment

* Python 3.11+
* CPU Only
* No GPU required
* No Internet during ranking
* Runtime under 5 minutes

---

# Large Files

The following files are not included because they exceed GitHub's file size limits:

* `candidates.jsonl`
* `candidate_embeddings.npy`

To regenerate them:

```bash
python embedder.py
```

Required inputs:

* candidates.jsonl
* Job Description

Generated artifacts:

* candidate_embeddings.npy
* jd_embedding.npy
* candidate_ids.json

---

# Output

The generated submission contains:

```
candidate_id
rank
score
reasoning
```

Example:

```
CAND_0076995,1,0.800127,
AI Research Engineer with 4.4 years of experience.
Demonstrates experience in Semantic Search and LlamaIndex,
aligning well with the job description.
Good recruiter engagement (73%).
```

---

# Reproducibility

To reproduce the final submission:

```bash
python submit_ranker.py
```

The script generates:

```
team_submission.csv
```

which passes the provided submission validator.

---

# Future Improvements

* Cross-encoder reranking for the top semantic candidates
* Better behavioral feature engineering
* Dynamic score calibration
* Improved reasoning generation using structured templates
* Domain-specific embedding models
