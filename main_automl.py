import json
import csv
from modules.retriever import HybridRetriever
from modules.evaluator import RAGEvaluator
from sentence_transformers import SentenceTransformer
from config import HP_GRID, PATHS

# --- FINAL GRID SETUP ---
# Wir verfeinern nur noch das Retrieval
HP_GRID_FINAL = {
    "top_k_papers": [15, 20, 25],
    "top_k_chunks": [10, 15, 20],
    "use_expansion": [False] # Wir wissen, False ist besser
}

model = SentenceTransformer('BAAI/bge-m3', device='cpu')
# WICHTIG: Nutze den Pfad zur 1024er DB
retriever = HybridRetriever(PATHS["db_1024"]) 

with open(PATHS["eval_set_1017"], 'r', encoding='utf-8') as f:
    eval_set = [json.loads(line) for line in f]
evaluator = RAGEvaluator(eval_set)

results = []

print("🚀 Starte Final AutoML Loop...")
for k_p in HP_GRID_FINAL["top_k_papers"]:
    for k_c in HP_GRID_FINAL["top_k_chunks"]:
        config = {
            "top_k_papers": k_p,
            "top_k_chunks": k_c,
            "use_expansion": False,
            "collection_name": "stage2_chunks_1024"
        }
        
        score = evaluator.run_benchmark(retriever, model, config)
        print(f"Test: Papers={k_p}, Chunks={k_c} -> Acc: {score:.2f}%")
        results.append({**config, "accuracy": score})

# Ergebnisse speichern
with open("final_optimization_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
