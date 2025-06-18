import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict

def load_model() -> SentenceTransformer:
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_embeddings(json_path: str) -> list:
    with open(json_path, 'r') as f:
        return json.load(f)

def embed_phrase(phrase: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode(phrase)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def find_closest_match(phrase: str, embedding_data: list, model: SentenceTransformer) -> Tuple[str, str, float]:
    phrase_emb = embed_phrase(phrase, model)
    
    best_id = None
    best_name = None
    best_score = -1

    for item in embedding_data:
        item_emb = np.array(item["Embedding"])
        score = cosine_similarity(phrase_emb, item_emb)
        if score > best_score:
            best_id = item["ID_Ra"]
            best_name = item["Name"]
            best_score = score

    return best_id, best_name, best_score

def find_top_n_matches(
    phrase: str,
    embedding_data: List[Dict],
    model: SentenceTransformer,
    n: int = 5
) -> List[Tuple[str, str, float]]:
    phrase_emb = embed_phrase(phrase, model)
    results = []

    for item in embedding_data:
        item_emb = np.array(item["Embedding"])
        score = cosine_similarity(phrase_emb, item_emb)
        results.append((item["ID_Ra"], item["Name"], score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:n]

def find_top_n_lower_emission_matches(
    phrase: str,
    embedding_data: List[Dict],
    model: SentenceTransformer,
    n: int = 5
) -> List[Tuple[str, str, float, float]]:
    phrase_emb = embed_phrase(phrase, model)

    all_scores = []
    print(embedding_data[:5])  # Debugging line to check the first few items in embedding_data
    for item in embedding_data:
        item_emb = np.array(item["Embedding"])
        score = cosine_similarity(phrase_emb, item_emb)
        all_scores.append({
            "ID_Ra": item["ID_Ra"],
            "Name": item["Name"],
            "Embedding": item_emb,
            "CO2": float(item.get("Total kg CO2-eq/kg", float("inf"))),
            "Score": score
        })


    all_scores.sort(key=lambda x: x["Score"], reverse=True)
    top_item = all_scores[0]
    co2_threshold = top_item["CO2"]

    lower_emission_items = [
        item for item in all_scores[1:] if item["CO2"] < co2_threshold
    ]

    print(f"Top item CO2: {co2_threshold:.2f} kg CO2-eq/kg")
    for item in lower_emission_items:
        print(f"Item ID: {item['ID_Ra']}, Name: {item['Name']}, CO2: {item['CO2']:.2f} kg CO2-eq/kg, Score: {item['Score']:.4f}")

    lower_emission_items.sort(key=lambda x: x["Score"], reverse=True)

    return [
        (item["ID_Ra"], item["Name"], item["Score"], item["CO2"])
        for item in lower_emission_items[:n]
    ]