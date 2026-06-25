import json
import pandas as pd
import xgboost as xgb
from datetime import datetime


def extract_features(candidate: dict) -> dict:
    """Flattens the redrob_signals object into numerical features for XGBoost."""
    
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

    features = {
        "profile_completeness": signals.get("profile_completeness_score", 0),
        "days_since_active": max(0, days_since_active),
        "open_to_work": int(signals.get("open_to_work_flag", False)),
        "response_rate": signals.get("recruiter_response_rate", 0.0),
        "avg_response_time": signals.get("avg_response_time_hours", 72),
        "connections": signals.get("connection_count", 0),
        "notice_period": signals.get("notice_period_days", 90),
        "github_score": signals.get("github_activity_score", -1),
        "interview_completion": signals.get("interview_completion_rate", 0.0),
        "linkedin_connected": int(signals.get("linkedin_connected", False))
    }

    return features


def train_behavioral_model(candidates_path: str):

    print("Loading candidates for XGBoost training...")

    with open("candidate_ids.json", "r", encoding="utf-8") as f:
        valid_ids = set(json.load(f))

    data = []

    # For candidates.jsonl
    with open(candidates_path, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:
                c = json.loads(line)

                if c.get("candidate_id") in valid_ids:
                    features = extract_features(c)
                    data.append(features)

            except Exception:
                continue

    df = pd.DataFrame(data)

    # Proxy label
    df["target"] = (
        (df["response_rate"] > 0.7)
        &
        (df["days_since_active"] < 30)
        &
        (df["interview_completion"] > 0.7)
    ).astype(int)

    X = df.drop(columns=["target"])
    y = df["target"]

    print(
        f"Training XGBoost on {len(X):,} profiles "
        f"(Positive rate: {y.mean():.2%})"
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X, y)

    model.save_model("behavioral_ranker.json")

    print("Model trained and saved to behavioral_ranker.json!")


if __name__ == "__main__":

    train_behavioral_model("candidates.jsonl")