# config.py
import os
import sys

IS_COLAB = 'google.colab' in sys.modules
IS_KAGGLE = os.path.exists('/kaggle')

# FIX FOR KAGGLE:
if IS_KAGGLE:
    # Kaggle-spezifische Pfade (angepasst für die Kaggle-Umgebung)
    BASE_MARKDOWN_DIR = "/kaggle/working/modulare-rag-pipeline/data/extracted_markdown"
    BASE_EVAL_FILE = "/kaggle/working/eval_set_100q.jsonl"
    BASE_CHECKPOINT_FILE = "/kaggle/working/indexing_checkpoint_1024.txt"
elif IS_COLAB:
    BASE_DATA_PATH = "/content/drive/MyDrive/rag_ml_data"
    BASE_MARKDOWN_DIR = os.path.join(BASE_DATA_PATH, "data/extracted_markdown")
    BASE_EVAL_FILE = "/content/eval_set_100q.jsonl"
    BASE_CHECKPOINT_FILE = os.path.join(BASE_DATA_PATH, "checkpoint_1024.txt")
else:
    # lokaler Laptop
    BASE_MARKDOWN_DIR = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown"
    BASE_EVAL_FILE = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_100q.jsonl"
    BASE_CHECKPOINT_FILE = r"C:\Users\ahmad\Desktop\rag_ml\data\papers\checkpoint_1024.txt"

DEVICE = "cuda" if (IS_COLAB or IS_KAGGLE) else "cpu"

# ... (Rest HP_GRID bleibt gleich) ...

PATHS = {
    "markdown": BASE_MARKDOWN_DIR,
    "eval_set": BASE_EVAL_FILE, # Einheitlicher Key für alle Skripte!
    "checkpoint_1024": BASE_CHECKPOINT_FILE,
    "results_csv": "final_optimization_results.csv" if (IS_COLAB or IS_KAGGLE) else r"C:\Users\ahmad\Desktop\rag_ml\final_optimization_results.csv"
}

# Hardware-Wahl
DEVICE = "cuda" if (IS_COLAB or IS_KAGGLE) else "cpu"

# 3. Das Hyperparameter Grid & Regeln für die Query Expansion
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
