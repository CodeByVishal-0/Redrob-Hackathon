# Redrob Hackathon - Hybrid AI Candidate Ranking System

## Overview

This project implements a hybrid candidate ranking system for the Redrob Hackathon.

The system combines:

- Semantic retrieval using Sentence Transformers
- Behavioral reranking using XGBoost
- Honeypot filtering
- Hybrid score fusion

The final output is the Top-100 ranked candidates.

---

## Repository Structure

```
.
├── candidates.jsonl
├── jd.txt
├── honeypot.py
├── embedder.py
├── train_behavior_model.py
├── submit_ranker.py
├── candidate_embeddings.npy
├── jd_embedding.npy
├── candidate_ids.json
├── behavioral_ranker.json
├── team_submission.csv
├── requirements.txt
└── submission_metadata.yaml
```

---

## Pipeline

### Step 1

Remove honeypots

```
python honeypot.py
```

---

### Step 2

Generate embeddings

```
python embedder.py
```

Produces

- candidate_embeddings.npy
- jd_embedding.npy
- candidate_ids.json

---

### Step 3

Train behavioral model

```
python train_behavior_model.py
```

Produces

```
behavioral_ranker.json
```

---

### Step 4

Generate submission

```
python submit_ranker.py
```

Produces

```
team_submission.csv
```

---

## Method

Hybrid Ranking

```
Final Score

=

0.75 × Semantic Similarity

+

0.25 × Behavioral Score
```

Semantic similarity uses cosine similarity between SentenceTransformer embeddings.

Behavioral score is predicted using XGBoost.

---

## Requirements

Python 3.11+

CPU Only

No internet required during ranking.

Runtime <5 minutes.