import sys
from sentence_transformers import SentenceTransformer
from modules.retriever import HybridRetriever
from modules.generator import ArxivGenerator

# 1. Setup
db_path = r"C:\Users\ahmad\Desktop\rag_ml\qdrant_db"
# Lade das Modell nur EINMAL
model = SentenceTransformer('BAAI/bge-m3', device='cpu')
retriever = HybridRetriever(db_path)
generator = ArxivGenerator(model_name="phi3:mini") # phi3:mini ist sicherer für 8GB RAM

# 2. Query
raw_query = "How to optimize LLM training?"

# Schritt 1: Anzeigen der Expansion (optional, nur zur Kontrolle)
expanded_query = retriever.expand_query(raw_query)
print(f"🔍 Original: {raw_query}")
print(f"🚀 Expanded: {expanded_query}")

# 3. Retrieve
# Wir übergeben die RAW_QUERY, da der Retriever intern expandiert und das Modell nutzt
print("Retrieving context...")
chunks = retriever.search(raw_query, model, top_k_papers=3, top_k_chunks=5, use_expansion=True)

# 4. Generate
if chunks:
    print("Generating answer...")
    # Nutze die raw_query oder expanded_query für den LLM-Prompt
    answer = generator.generate_answer(raw_query, chunks)
    print("\n=== FINAL ANSWER ===\n")
    print(answer)
else:
    print("No relevant papers found.")

# Sauber beenden (Wichtig für msvcrt Fehler)
del retriever
sys.exit(0)
