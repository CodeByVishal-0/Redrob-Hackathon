import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(candidate):

    signals = candidate.get("redrob_signals", {})

    reference_date = datetime(2024, 1, 1)

    last_active = signals.get("last_active_date")

    try:
        if last_active:
            days_since_active = (
                reference_date -
                datetime.strptime(last_active, "%Y-%m-%d")
            ).days
        else:
            days_since_active = 365

    except Exception:
        days_since_active = 365

    return {
        "profile_completeness":
            signals.get("profile_completeness_score", 0),

        "days_since_active":
            max(0, days_since_active),

        "open_to_work":
            int(signals.get("open_to_work_flag", False)),

        "response_rate":
            signals.get("recruiter_response_rate", 0),

        "avg_response_time":
            signals.get("avg_response_time_hours", 72),

        "connections":
            signals.get("connection_count", 0),

        "notice_period":
            signals.get("notice_period_days", 90),

        "github_score":
            signals.get("github_activity_score", 0),

        "interview_completion":
            signals.get("interview_completion_rate", 0),

        "linkedin_connected":
            int(signals.get("linkedin_connected", False))
    }


# ==========================================================
# COSINE SIMILARITY
# ==========================================================
def calculate_cosine_similarity(query_vec, matrix):

    dot_product = np.dot(matrix, query_vec)

    query_norm = np.linalg.norm(query_vec)

    matrix_norm = np.linalg.norm(matrix, axis=1)

    similarities = dot_product / (query_norm * matrix_norm)

    return similarities


# ==========================================================
# REASONING
# ==========================================================
def generate_reasoning(candidate):

    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Professional")
    years = profile.get("years_of_experience", 0)

    response_rate = signals.get(
        "recruiter_response_rate",
        0
    )

    notice = signals.get(
        "notice_period_days",
        90
    )

    skill_names = [
        s.get("name", "").lower()
        for s in skills
    ]

    retrieval = [
        s for s in [
            "recommendation systems",
            "semantic search",
            "embeddings",
            "sentence transformers",
            "langchain",
            "llamaindex",
            "learning to rank",
            "faiss",
            "qdrant",
            "milvus",
            "weaviate",
            "pinecone",
            "haystack"
        ]
        if s in skill_names
    ]

    llm = [
        s for s in [
            "llms",
            "prompt engineering",
            "transformers",
            "hugging face transformers",
            "rag"
        ]
        if s in skill_names
    ]

    ml = [
        s for s in [
            "python",
            "tensorflow",
            "pytorch",
            "scikit-learn",
            "computer vision",
            "nlp",
            "speech recognition",
            "reinforcement learning",
            "feature engineering"
        ]
        if s in skill_names
    ]

    strengths = []

    if retrieval:
        strengths.append(
            ", ".join(retrieval[:2])
        )

    if llm:
        strengths.append(
            ", ".join(llm[:2])
        )

    if not strengths and ml:
        strengths.append(
            ", ".join(ml[:2])
        )

    reason = (
        f"{title} with "
        f"{years:.1f} years of experience. "
    )

    if strengths:
        reason += (
            f"Demonstrates experience in "
            f"{' and '.join(strengths)}, "
            f"aligning well with the JD. "
        )

    if response_rate >= 0.80:
        reason += (
            f"High recruiter engagement "
            f"({response_rate:.0%}). "
        )
    elif response_rate >= 0.60:
        reason += (
            f"Good recruiter engagement "
            f"({response_rate:.0%}). "
        )
    else:
        reason += (
            f"Moderate recruiter engagement "
            f"({response_rate:.0%}). "
        )

    if notice > 90:
        reason += (
            f"Long notice period "
            f"({notice} days) may delay onboarding."
        )

    return reason.strip()


# ==========================================================
# MAIN
# ==========================================================
def run_online_ranking():

    start_time = time.time()

    print("Loading artifacts...")

    candidate_embeddings = np.load(
        "candidate_embeddings.npy"
    )

    jd_embedding = np.load(
        "jd_embedding.npy"
    )

    with open(
        "candidate_ids.json",
        "r",
        encoding="utf-8"
    ) as f:
        candidate_ids = json.load(f)

    # Load candidate dictionary
    print("Loading candidate profiles...")

    candidate_lookup = {}

    valid_ids = set(candidate_ids)

    with open(
        "candidates.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            candidate = json.loads(line)

            if candidate["candidate_id"] in valid_ids:
                candidate_lookup[
                    candidate["candidate_id"]
                ] = candidate

    # Load behavioral model
    behavior_model = xgb.XGBClassifier()

    behavior_model.load_model(
        "behavioral_ranker.json"
    )

    # ======================================================
    # STAGE 1 : SEMANTIC SEARCH
    # ======================================================
    print("Computing semantic similarity...")

    semantic_scores = calculate_cosine_similarity(
        jd_embedding,
        candidate_embeddings
    )

    top_indices = np.argsort(
        semantic_scores
    )[-2000:][::-1]

    print(
        "Top semantic candidates:",
        len(top_indices)
    )

    # Feature order used during training
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

    # ======================================================
    # STAGE 2 : HYBRID RE-RANKING
    # ======================================================
    results = []

    for idx in top_indices:

        candidate_id = candidate_ids[idx]

        candidate = candidate_lookup.get(
            candidate_id
        )

        if candidate is None:
            continue

        features = extract_features(candidate)

        X = pd.DataFrame(
            [features],
            columns=feature_order
        )

        behavioral_score = (
            behavior_model
            .predict_proba(X)[0][1]
        )

        semantic_score = float(
            semantic_scores[idx]
        )

        final_score = (
            0.75 * semantic_score
            + 0.25 * behavioral_score
        )

        reasoning = generate_reasoning(
            candidate
        )

        results.append(
            {
                "candidate_id":
                    candidate_id,

                "score":
                    round(final_score, 8),

                "reasoning":
                    reasoning
            }
        )

    # ======================================================
    # TOP 100
    # ======================================================
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    top_100 = results[:100]

    df = pd.DataFrame(top_100)

    df["rank"] = range(
        1,
        len(df) + 1
    )

    # IMPORTANT: order required by validator
    df = df[
        [
            "candidate_id",
            "rank",
            "score",
            "reasoning"
        ]
    ]

    df.to_csv(
        "team_submission.csv",
        index=False,
        encoding="utf-8"
    )

    print()
    print("===================================")
    print("Top 100 candidates saved.")
    print("Output file : team_submission.csv")
    print(
        f"Completed in {time.time()-start_time:.2f} sec"
    )
    print("===================================")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":

    run_online_ranking()
