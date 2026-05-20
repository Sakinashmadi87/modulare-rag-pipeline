#main_automl.py
import json
import csv
import os
import sys
import torch
from sentence_transformers import SentenceTransformer
from modules.evaluator import RAGEvaluator

# Füge aktuellen Pfad hinzu, damit Module sicher gefunden werden
sys.path.append(os.getcwd())
from config import HP_GRID, PATHS, IS_COLAB, IS_KAGGLE

# --- FINAL GRID SETUP ---
HP_GRID_FINAL = {
    "top_k_papers": [15, 20, 25],
    "top_k_chunks": [10, 15, 20],
    "use_expansion": [False]
}

def get_cloud_credentials():
    """Liest die Qdrant-Zugangsdaten plattformunabhängig aus."""
    url, api_key = None, None
    
    if IS_COLAB:
        try:
            from google.colab import userdata
            url = userdata.get('QDRANT_URL')
            api_key = userdata.get('QDRANT_API_KEY')
            print("🌐 Umgebung erkannt: Google Colab")
        except Exception:
            pass
            
    elif IS_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            url = user_secrets.get_secret("QDRANT_URL")
            api_key = user_secrets.get_secret("QDRANT_API_KEY")
            print("🌐 Umgebung erkannt: Kaggle")
        except Exception:
            pass
            
    # Fallback für Umgebungsvariablen (falls in Colab/Kaggle Secrets nicht gesetzt)
    url = url or os.getenv('QDRANT_URL')
    api_key = api_key or os.getenv('QDRANT_API_KEY')
    
    return url, api_key

def main():
    # 1. Hardware-Erkennung (GPU vs. CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Initialisiere Embedding-Modell (BGE-M3) auf: {device.upper()}")
    model = SentenceTransformer('BAAI/bge-m3', device=device)

    # 2. Retriever dynamisch laden
    # Erst jetzt importieren, damit sys.path Anpassungen greifen
    from modules.retriever import HybridRetriever
    
    qdrant_url, qdrant_key = get_cloud_credentials()
    
    if qdrant_url and qdrant_key:
        print("🌐 Verbinde mit Qdrant Cloud Cluster...")
        retriever = HybridRetriever(url=qdrant_url, api_key=qdrant_key)
    else:
        print("🏠 Keine Cloud-Logins gefunden. Verbinde mit lokaler Qdrant-Datenbank...")
        # Nutzt den im Retriever hartcodierten Standard-Pfad
        retriever = HybridRetriever()

    # 3. Lade Evaluationsset (eval_set_100q.jsonl) - Pfad wird automatisch je nach Umgebung angepasst
   
    eval_path = PATHS.get("eval_set", os.path.join(os.getcwd(), "eval_set_100q.jsonl"))
    print(f"📂 Lade Evaluationsset aus: {eval_path}")

    
    if not os.path.exists(eval_path):
        print(f"❌ Fehler: Datei {eval_path} existiert nicht!")
        return

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_set = [json.loads(line) for line in f]
    evaluator = RAGEvaluator(eval_set)

    results = []

    # 4. Der AutoML Loop
    print(f"🚀 Starte Final AutoML Loop für {len(eval_set)} Fragen...")
    for k_p in HP_GRID_FINAL["top_k_papers"]:
        for k_c in HP_GRID_FINAL["top_k_chunks"]:
            config = {
                "top_k_papers": k_p,
                "top_k_chunks": k_c,
                "use_expansion": False,
                "collection_name": "stage2_chunks_1024"
            }
            
            score = evaluator.run_benchmark(retriever, model, config)
            print(f"📊 Grid: Papers={k_p}, Chunks={k_c} -> Accuracy: {score:.2f}%")
            results.append({**config, "accuracy": score})

    # 5. Speicherort-Verwaltung ( results_csv aus config oder lokal im Arbeitsverzeichnis)
    results_file = PATHS.get("results_csv", os.path.join(os.getcwd(), "final_optimization_results.csv"))
    print(f"💾 Speichere Ergebnisse in: {results_file}")
    
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("✅ AutoML-Lauf erfolgreich beendet!")

if __name__ == "__main__":
    main()
