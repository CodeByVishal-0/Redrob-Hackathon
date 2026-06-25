import gzip
import json
import os
from typing import List, Dict


class HoneypotDetector:
    """
    Identifies logically impossible profiles to avoid Stage 3 disqualification.
    """

    @staticmethod
    def is_honeypot(candidate: Dict) -> bool:
        # Check 1: Expert proficiency with 0 years of use
        skills = candidate.get("skills", [])
        for skill in skills:
            if (
                skill.get("proficiency") == "expert"
                and skill.get("years_used", 0) == 0
            ):
                return True

        # Check 2: Experience duration exceeds company age
        experience = candidate.get("experience", [])
        for job in experience:
            years_at_job = job.get("duration_years", 0)
            company_age = job.get("company_age_years", 999)

            if years_at_job > company_age:
                return True

        return False


def load_and_clean_candidates(filepath: str) -> List[Dict]:
    """
    Loads candidates from a JSONL or JSONL.GZ file
    and filters out honeypot profiles.
    """

    clean_candidates = []
    trap_count = 0

    print(f"Loading candidates from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {os.path.abspath(filepath)}"
        )

    # Choose correct file opener
    opener = gzip.open if filepath.endswith(".gz") else open

    try:
        with opener(filepath, "rt", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):

                if not line.strip():
                    continue

                try:
                    candidate = json.loads(line)

                    if HoneypotDetector.is_honeypot(candidate):
                        trap_count += 1
                        continue

                    clean_candidates.append(candidate)

                except json.JSONDecodeError:
                    print(
                        f"Warning: Invalid JSON at line {line_number}. Skipping."
                    )

    except Exception as e:
        print(f"Error while reading file: {e}")
        return []

    print("\n===== SUMMARY =====")
    print(f"Valid candidates loaded : {len(clean_candidates)}")
    print(f"Honeypots removed       : {trap_count}")
    print("=====================\n")

    return clean_candidates


if __name__ == "__main__":

    # Change this path if needed
    data_path = "candidates.jsonl"

    print("Starting candidate loader...\n")

    valid_talent_pool = load_and_clean_candidates(data_path)

    if valid_talent_pool:
        print("First valid candidate:\n")
        print(
            json.dumps(
                valid_talent_pool[0],
                indent=2
            )[:1000]
        )