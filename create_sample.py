import json
import pandas as pd

# Load your official submission
submission = pd.read_csv("team_submission.csv")
top_ids = set(submission["candidate_id"])

count = 0

with open("candidates.jsonl", "r", encoding="utf-8") as fin, \
     open("demo/sample_candidates.jsonl", "w", encoding="utf-8") as fout:

    for line in fin:
        candidate = json.loads(line)

        if candidate["candidate_id"] in top_ids:
            fout.write(json.dumps(candidate) + "\n")
            count += 1

print(f"Saved {count} candidates.")