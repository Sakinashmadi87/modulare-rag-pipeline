import json
import csv
import os
from modules.retriever import HybridRetriever
from modules.evaluator import RAGEvaluator
from sentence_transformers import SentenceTransformer
from config import HP_GRID, PATHS, IS_COLAB

# --- FINAL GRID SETUP ---
HP_GRID_FINAL = {
    "top_k_papers": [15, 20, 25],
    "top_k_chunks": [10, 15, 20],
    "use_expansion": [False]
}

# 1. Modell auf das richtige Device laden (GPU in Colab, CPU lokal)
device = "cuda" if IS_COLAB else "cpu"
print(f"🖥️ Initialisiere Embedding-Modell auf: {device.upper()}")
model = SentenceTransformer('BAAI/bge-m3', device=device)

# 2. Qdrant-Verbindung initialisieren
if IS_COLAB:
    from google.colab import userdata
    qdrant_url = userdata.get('QDRANT_URL')
    qdrant_api_key = userdata.get('QDRANT_API_KEY')
    
    print("🌐 Verbinde mit Qdrant Cloud Cluster...")
    # Hier übergeben wir die Cloud-Verbindungsdaten an Ihren Retriever
    retriever = HybridRetriever(location=qdrant_url, api_key=qdrant_api_key)
else:
    print("🏠 Verbinde mit lokaler Qdrant-Datenbank...")
    LOCAL_DB_1024 = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024"
    retriever = HybridRetriever(LOCAL_DB_1024)

# 3. Evaluationsdaten laden
print(f"📂 Lade Evaluationsset aus: {PATHS['eval_set_1017']}")
with open(PATHS["eval_set_1017"], 'r', encoding='utf-8') as f:
    eval_set = [json.loads(line) for line in f]
evaluator = RAGEvaluator(eval_set)

results = []

# 4. Der AutoML Loop
print("🚀 Starte Final AutoML Loop in der Cloud...")
for k_p in HP_GRID_FINAL["top_k_papers"]:
    for k_c in HP_GRID_FINAL["top_k_chunks"]:
        config = {
            "top_k_papers": k_p,
            "top_k_chunks": k_c,
            "use_expansion": False,
            "collection_name": "stage2_chunks_1024"  # Name der Collection in Qdrant
        }
        
        score = evaluator.run_benchmark(retriever, model, config)
        print(f"Test: Papers={k_p}, Chunks={k_c} -> Acc: {score:.2f}%")
        results.append({**config, "accuracy": score})

# 5. Ergebnisse direkt im Google Drive (oder lokal) speichern
results_path = PATHS["results_csv"]
print(f"💾 Speichere Ergebnisse in: {results_path}")
with open(results_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("✅ AutoML erfolgreich beendet!")
