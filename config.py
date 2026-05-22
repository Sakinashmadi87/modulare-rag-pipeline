# config.py
import os
import sys

# === 1. Umgebungs-Identifikation ===
IS_COLAB = 'google.colab' in sys.modules
IS_KAGGLE = os.path.exists('/kaggle')

# === 2. Umgebungsabhängige Basis-Pfade ===
if IS_KAGGLE:
    BASE_DATA_ROOT = "/kaggle/input/datasets/sakinaahmadi/rag-ml-data"
    BASE_OUTPUT_ROOT = "/kaggle/working"
    BASE_EVAL_FILE = "/kaggle/input/datasets/sakinaahmadi/rag-automl-gold-standard-100q/eval_set_100q.jsonl"
elif IS_COLAB:
    BASE_DATA_ROOT = "/content/drive/MyDrive/rag_ml_data"
    BASE_OUTPUT_ROOT = "/content"
    BASE_EVAL_FILE = "/content/eval_set_100q.jsonl"
else:
    # Lokaler Laptop
    BASE_DATA_ROOT = r"C:\Users\ahmad\Desktop\rag_ml\data"
    BASE_OUTPUT_ROOT = r"C:\Users\ahmad\Desktop\rag_ml"
    BASE_EVAL_FILE = os.path.join(BASE_DATA_ROOT, "papers", "eval_set_100q.jsonl")

# === 3. Dynamische Pfad-Generierung ===
PATHS = {
    "markdown": os.path.join(BASE_DATA_ROOT, "papers", "extracted_markdown"),
    "eval_set": BASE_EVAL_FILE,
    "checkpoint_1024": os.path.join(BASE_OUTPUT_ROOT, "indexing_checkpoint_1024.txt"),
    "results_csv": os.path.join(BASE_OUTPUT_ROOT, "final_optimization_results.csv"),
    "output_root": BASE_OUTPUT_ROOT,
    "data_root": BASE_DATA_ROOT,
}

# === 4. Hardware & Device ===
DEVICE = "cuda" if (IS_COLAB or IS_KAGGLE) else "cpu"

# === 5. Query Expansion Regeln (für RAG-Optimierung) ===
EXPANSION_RULES = {
    "scaling laws": "compute-optimal Chinchilla model size parameters",
    "rag": "retrieval-augmented generation knowledge retrieval",
    "llm": "large language models transformer architecture",
    "code": "programming syntax software engineering reasoning",
    "diffusion": "generative denoising probabilistic modeling",
    "evaluation": "benchmarking metrics ragas faithfulness"
}

# === 6. Hyperparameter Grid (flexibel, kann später mit `itertools.product` genutzt werden) ===
HP_GRID = {
    "top_k_papers": [5, 10, 15, 20],
    "top_k_chunks": [5, 10, 15],
    "use_expansion": [False, True],  # ✅ Jetzt flexibel!
    "chunk_size": [1024],
    "chunk_overlap": [100],
    "math_heuristic": [True],
    "embedding_model": ["BAAI/bge-m3"],
    "llm_model": ["unsloth/llama-3-8b-Instruct"],
    "device": [DEVICE],  # ✅ Dynamisch, aber nur ein Wert
}

# === 7. Optional: Debug-Info für Logging ===
def get_env_info():
    return {
        "is_colab": IS_COLAB,
        "is_kaggle": IS_KAGGLE,
        "device": DEVICE,
        "data_root": PATHS["data_root"],
        "output_root": PATHS["output_root"],
    }