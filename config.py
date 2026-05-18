import os
import sys


# Erkennt automatisch, ob der Code in Colab oder Kaggle läuft
IS_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IS_KAGGLE = os.path.exists('/kaggle')

if IS_COLAB:
    BASE_PATH = "/content/drive/MyDrive/rag_ml_data"
    DEVICE = "cuda"
    print("🤖 [CONFIG] Google Colab erkannt! Nutze GPU und Google Drive.")
elif IS_KAGGLE:
    # Auf Kaggle binden wir Google Drive als Webdav oder ein Kaggle Dataset ein.
    # Für den Moment setzen wir den Pfad auf den temporären Arbeitsordner von Kaggle.
    BASE_PATH = "/kaggle/working/rag_ml_data"
    DEVICE = "cuda"  # Zündet die kostenlose T4 GPU auf Kaggle!
    print("🦅 [CONFIG] Kaggle Umgebung erkannt! Nutze Kaggle T4 GPU.")
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
