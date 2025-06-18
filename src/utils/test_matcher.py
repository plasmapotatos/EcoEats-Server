from embedding_matcher import load_model, load_embeddings, find_top_n_lower_emission_matches

if __name__ == "__main__":
    model = load_model()
    data = load_embeddings("data/embeddings_output.json")

    phrase = "cheddar cheese"
    results = find_top_n_lower_emission_matches(phrase, data, model, n=5)

    print("Lower-emission matches:")
    for i, (id_ra, name, score, co2) in enumerate(results, 1):
        print(f"{i}. ID_Ra: {id_ra}, Name: {name}, Similarity: {score:.4f}, CO2-eq/kg: {co2:.2f}")