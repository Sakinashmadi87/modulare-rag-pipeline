# config.py

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
    # --- Focus on Breadth to break the 38% limit ---
    "top_k_papers": [5, 10, 15, 20],  # Increased to find more missing papers
    "top_k_chunks": [5, 10, 15],      # Increased for deeper context
    "use_expansion": [False],         # Locked to False as it performed better

    # --- Information for your records ---
    "chunk_size": [1024],             # Your winner for efficiency
    "chunk_overlap": [100],
    "math_heuristic": [True],         
    
    # --- Constants ---
    "embedding_model": "BAAI/bge-m3",
    "llm_model": "phi3:mini",
    "device": "cpu"                  
}
# 3. Pfade 
PATHS = {
    "db": r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db",
    "markdown": r"C:\Users\ahmad\Desktop\rag_ml\data\papers\extracted_markdown",
    "db_1024": r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db_1024",
    "eval_set": r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_new.jsonl",
    "eval_set_1017": r"C:\Users\ahmad\Desktop\rag_ml\data\papers\eval_set_1017.jsonl" 
}
