import os
import sys

# 100% sichere Erkennung für Google Colab
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')

if IS_COLAB:
    # Falls der Ordner in Ihrem Google Drive direkt 'rag_ml_data' heißt:
    BASE_PATH = "/content/drive/MyDrive/rag_ml_data"
    DEVICE = "cuda"  # Schaltet JETZT die schnelle Cloud-GPU ein
    print("🤖 [CONFIG] Google Colab erkannt! Nutze GPU und Google Drive.")
else:
    BASE_PATH = r"C:\Users\ahmad\Desktop\rag_ml"
    DEVICE = "cpu"
    print("🏠 [CONFIG] Lokale Umgebung erkannt! Nutze CPU.")

# 1. Die Regeln für die Query Expansion
EXPANSION_RULES = {
    "scaling laws": "compute-optimal Chinchilla model size parameters",
    "rag": "retrieval-augmented generation knowledge retrieval",
    "llm": "large language models transformer architecture",
    "code": "programming syntax software engineering reasoning",
    "diffusion": "generative denoising probabilistic modeling",
    "evaluation": "benchmarking metrics ragas faithfulness"
}

# 2. Das AutoML Hyperparameter Grid
HP_GRID = {
    "top_k_papers": [5, 10, 15, 20],
    "top_k_chunks": [5, 10, 15],
    "use_expansion": [False],
    "chunk_size": [1024],
    "chunk_overlap": [100],
    "math_heuristic": [True],         
    "embedding_model": "BAAI/bge-m3",
    "llm_model": "phi3:mini",
    "device": DEVICE                  
}

# 3. Pfade dynamisch generiert
PATHS = {
    "markdown": os.path.join(BASE_PATH, "data", "papers", "extracted_markdown"),
    "eval_set": os.path.join(BASE_PATH, "data", "papers", "eval_set_new.jsonl"),
    "eval_set_1017": os.path.join(BASE_PATH, "data", "papers", "eval_set_1017.jsonl"),
    "results_csv": os.path.join(BASE_PATH, "final_optimization_results.csv")
}
