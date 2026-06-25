import json
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from honeypot import load_and_clean_candidates

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

    # Skills
    skill_text = []

    for skill in skills:
        name = skill.get("name", "")
        months = skill.get("duration_months", 0)
        skill_text.append(
            f"{name} ({months} months)"
        )

    # Certifications
    cert_text = []

    for cert in certs:
        cert_text.append(
            cert.get("name", "")
        )

    # Work history
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

    # Derived tags
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

def generate_and_save_embeddings(clean_candidates: list, jd_text: str):
    """
    Generates vectors for all candidates and the JD, then saves them to disk.
    """
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    # This model is ~90MB, very fast on CPU, and produces 384-dimensional vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Embed the Job Description
    print("Embedding Job Description...")
    jd_embedding = model.encode([jd_text])[0]
    np.save("jd_embedding.npy", jd_embedding)
    
    # 2. Format and Embed Candidates
    print(f"Formatting {len(clean_candidates)} candidate profiles...")
    candidate_docs = [build_candidate_document(c) for c in clean_candidates]
    candidate_ids = [c.get('candidate_id') for c in clean_candidates]
    
    print("Generating candidate embeddings (this will take a few minutes)...")
    start_time = time.time()
    
    # Encode in batches for memory efficiency
    candidate_embeddings = model.encode(candidate_docs, batch_size=256, show_progress_bar=True)
    
    print(f"Finished embedding in {time.time() - start_time:.2f} seconds.")
    
    # 3. Save artifacts to disk for Phase 2
    print("Saving embeddings and ID mapping to disk...")
    np.save("candidate_embeddings.npy", candidate_embeddings)
    
    with open("candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)
        
    print("Offline pre-computation complete! Artifacts saved.")

# --- Execution (Continuing from previous script) ---
# Assuming `valid_talent_pool` is the list returned from our HoneypotDetector
valid_talent_pool = load_and_clean_candidates("candidates.jsonl")

jd_text = "Senior AI Engineer. 5-9 years experience. Modern ML systems, embeddings, retrieval, ranking, LLMs. Product-engineering attitude. Shipped end-to-end ranking system. Python. Vector databases."
generate_and_save_embeddings(valid_talent_pool, jd_text)