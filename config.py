import os
import sys

# 1. Automatische Erkennung der Umgebung
IS_COLAB = 'google.colab' in sys.modules
IS_KAGGLE = os.path.exists('/kaggle')

# 2. Dynamische Pfad-Zuweisung über Umgebungsvariablen 
# Wenn eine Variable nicht im System existiert, nimmt Python den Standardpfad (r"C:\...")
BASE_MARKDOWN_DIR = os.getenv("RAG_MARKDOWN_DIR", r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown")
BASE_EVAL_FILE = os.getenv("RAG_EVAL_FILE", r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_100q.jsonl")
BASE_CHECKPOINT_FILE = os.getenv("RAG_CHECKPOINT_FILE", r"C:\Users\ahmad\Desktop\rag_ml\data\papers\checkpoint_1024.txt")

# Hardware-Wahl
DEVICE = "cuda" if (IS_COLAB or IS_KAGGLE) else "cpu"

# 3. Das Hyperparameter Grid & Regeln (Unverändert)
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
    "llm_model": "unsloth/llama-3-8b-Instruct",  # <-- UPDATED FOR CLOUD PRODUCTION
    "device": DEVICE                  
}

PATHS = {
    "markdown": BASE_MARKDOWN_DIR,
    "eval_set": "/kaggle/working/eval_set_100q.jsonl" if IS_KAGGLE else ("/content/eval_set_100q.jsonl" if IS_COLAB else BASE_EVAL_FILE),
    "checkpoint_1024": BASE_CHECKPOINT_FILE,
    "results_csv": "final_optimization_results.csv" if (IS_COLAB or IS_KAGGLE) else r"C:\Users\ahmad\Desktop\rag_ml\final_optimization_results.csv"
}
