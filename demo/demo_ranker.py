import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
from submit_ranker import (
    extract_features,
    calculate_cosine_similarity,
    generate_reasoning
)

def build_candidate_document(candidate: dict) -> str:
    """
    Rich semantic document for retrieval.
    """

    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])

    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    company_size = profile.get("current_company_size", "")
    industry = profile.get("current_industry", "")
    years_exp = profile.get("years_of_experience", 0)

    headline = profile.get("headline", "")
    summary = profile.get("summary", "")

    skill_text = []

    for skill in skills:
        name = skill.get("name", "")
        months = skill.get("duration_months", 0)

        skill_text.append(
            f"{name} ({months} months)"
        )

    cert_text = []

    for cert in certs:
        cert_text.append(
            cert.get("name", "")
        )

    experience_text = []

    for job in history[:4]:

        title = job.get("title", "")
        company = job.get("company", "")
        description = job.get("description", "")
        industry_name = job.get("industry", "")

        experience_text.append(
            f"{title} at {company}. "
            f"Industry {industry_name}. "
            f"{description}"
        )

    all_text = str(candidate).lower()

    tags = []

    vector_terms = [
        "faiss",
        "qdrant",
        "milvus",
        "weaviate",
        "pinecone",
        "elasticsearch"
    ]

    retrieval_terms = [
        "retrieval",
        "rag",
        "ranking",
        "learning to rank",
        "recommendation",
        "embedding",
        "sentence transformers",
        "haystack"
    ]

    llm_terms = [
        "llm",
        "prompt engineering",
        "transformers",
        "hugging face"
    ]

    mlops_terms = [
        "mlflow",
        "kubeflow",
        "airflow",
        "bentoml"
    ]

    if any(term in all_text for term in vector_terms):
        tags.append("vector_database")

    if any(term in all_text for term in retrieval_terms):
        tags.append("retrieval_systems")

    if any(term in all_text for term in llm_terms):
        tags.append("llm")

    if any(term in all_text for term in mlops_terms):
        tags.append("mlops")

    document = f"""
Current title:
{current_title}

Current company:
{current_company}

Company size:
{company_size}

Industry:
{industry}

Years of experience:
{years_exp}

Headline:
{headline}

Professional summary:
{summary}

Skills:
{", ".join(skill_text)}

Certifications:
{", ".join(cert_text)}

Experience:
{" ".join(experience_text)}

Derived tags:
{" ".join(tags)}
"""

    return document


def load_candidates():
    candidates = []

    with open(
        os.path.join(BASE_DIR, "sample_candidates.jsonl"),
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    return candidates


def main():

    start = time.time()

    print("Loading sample candidates...")

    candidates = load_candidates()

    print(f"Loaded {len(candidates)} candidates")

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Building semantic documents...")

    docs = [
        build_candidate_document(c)
        for c in candidates
    ]

    print("Generating embeddings...")

    candidate_embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=True
    )

    with open(
        os.path.join(BASE_DIR, "jd.txt"),
        "r",
        encoding="utf-8"
    ) as f:
        jd_text = f.read()

    jd_embedding = model.encode([jd_text])[0]

    semantic_scores = calculate_cosine_similarity(
        jd_embedding,
        candidate_embeddings
    )

    behavior_model = xgb.XGBClassifier()
    behavior_model.load_model(
        os.path.join(ROOT_DIR, "behavioral_ranker.json")
    )

    feature_order = [
        "profile_completeness",
        "days_since_active",
        "open_to_work",
        "response_rate",
        "avg_response_time",
        "connections",
        "notice_period",
        "github_score",
        "interview_completion",
        "linkedin_connected"
    ]

    results = []

    for i, candidate in enumerate(candidates):

        features = extract_features(candidate)

        X = pd.DataFrame(
            [features],
            columns=feature_order
        )

        behavioral = behavior_model.predict_proba(X)[0][1]

        semantic = float(semantic_scores[i])

        score = (
            0.75 * semantic
            + 0.25 * behavioral
        )

        results.append({
            "candidate_id":
                candidate["candidate_id"],
            "score":
                round(score,8),
            "reasoning":
                generate_reasoning(candidate)
        })

    results = sorted(
        results,
        key=lambda x:x["score"],
        reverse=True
    )

    df = pd.DataFrame(results)

    df["rank"] = range(
        1,
        len(df)+1
    )

    df = df[
        [
            "candidate_id",
            "rank",
            "score",
            "reasoning"
        ]
    ]

    output_file = os.path.join(
    BASE_DIR,
    "team_submission.csv"
)

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(df.head())

    print()

    print("=" * 50)
    print("Demo completed successfully!")
    print(f"Candidates ranked : {len(df)}")
    print(f"Output CSV        : {output_file}")
    print(f"Completed in      : {time.time()-start:.2f} sec")
    print("=" * 50)


if __name__ == "__main__":
    main()