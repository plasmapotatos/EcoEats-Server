import argparse
import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/carbonData.csv", help="Path to the input CSV file")
    parser.add_argument("--output", default="data/embeddings_output.json", help="Path to the output JSON file")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"The specified CSV file does not exist: {args.csv_path}")

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output)) 

    df = pd.read_csv(args.csv_path)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    ids = df["ID_Ra"].tolist()
    names = df["Name"].astype(str).tolist()
    embeddings = model.encode(names, show_progress_bar=True)

    data = [{"ID_Ra": id_ra, "Name": name, "Embedding": emb.tolist()} for id_ra, name, emb in zip(ids, names, embeddings)]

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
