from pipeline.pinecone_pipeline import PineconePipeline
from pathlib import Path
import json
import csv

def convert_csv_to_json(csv_path: str, output_json_path: str, id_prefix: str = "resep-"):
    """
    Convert a CSV dataset into JSON format for embedding/upsert.

    Expected CSV columns:
        id, category, url, title, image, ingredients, steps, all_text

    - ingredients can be either string or list-like
    - steps can be text or a JSON-like string [{"text": "..."}]
    """
    data = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                recipe_id = row.get("id") or f"{id_prefix}{i}"

                # Safely parse ingredients and steps if needed
                ingredients = row.get("ingredients")
                if ingredients:
                    try:
                        ingredients = json.loads(ingredients)
                    except Exception:
                        ingredients = [ing.strip() for ing in ingredients.split(",") if ing.strip()]

                steps = row.get("steps")
                if steps:
                    try:
                        steps = json.loads(steps)
                    except Exception:
                        steps = [{"text": steps}]

                item = {
                    "id": recipe_id,
                    "category": row.get("category"),
                    "url": row.get("url"),
                    "title": row.get("title"),
                    "image": row.get("image"),
                    "ingredients": ingredients,
                    "steps": steps,
                    # You can concatenate title + ingredients + steps as "all_text"
                    "all_text": row.get("all_text")
                        or f"{row.get('title', '')}. {row.get('ingredients', '')}. {row.get('steps', '')}",
                    "text": row.get("ingredients") or ""
                }

                data.append(item)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Converted {len(data)} rows from {csv_path} → {output_json_path}")

    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
    except Exception as e:
        print(f"⚠️ Error converting CSV to JSON: {e}")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FOLDER = BASE_DIR / "data" / "clean"

if __name__ == "__main__":
    pipe = PineconePipeline("pipeline/model/bm25_params.json")

    # csv_path = "data/raw/resep_ayam.csv"
    # output_json = "data/clean/resep_ayam.json"
    # convert_csv_to_json(csv_path, output_json)

    # pipe.create_indexes()

    # dense_count, sparse_count = pipe.generate_and_upsert_from_file(output_json)
    # print(f"Done. {dense_count} dense and {sparse_count} sparse vectors inserted.")

    query = "resep ayam goreng lengkuas"

    results = pipe.search_and_fetch_full(query, top_k=5)

    # Make output folder if not exists
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    output_path = output_dir / f"results_{query.replace(' ', '_')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {output_path}")